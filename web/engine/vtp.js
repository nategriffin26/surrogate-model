/**
 * Reads a VTK XML PolyData file (.vtp) into a `ParsedSurface`, streaming: the file is read in slices
 * and base64 and zlib are decoded straight into typed arrays. Arrays the engine does not use are
 * stepped over without decoding (read past when inline, never read when appended), and reading
 * stops once every needed array is in, so a DrivAerML file's trailing solver fields are never read.
 *
 * Handles inline binary (base64), appended raw and appended base64 data, zlib compression, UInt32
 * and UInt64 headers, ASCII arrays, and polygons of any size. Anything else is rejected with an Error
 * that says what is wrong and what to do about it.
 *
 * Also writes such a file back with cell arrays added (`addCellArrays`), composed from slices of the
 * original rather than a copy of it.
 */

/** The CFD pressure coefficient; the only solver array the engine reads, and only for comparison. */
export const CFD_ARRAY = 'CpMeanTrim';

const SLICE_BYTES = 8 << 20;
const latin1 = new TextDecoder('latin1');

const SCALAR_BYTES = {
  Int8: 1, UInt8: 1, Int16: 2, UInt16: 2, Int32: 4, UInt32: 4, Int64: 8, UInt64: 8, Float32: 4, Float64: 8,
};

const RESAVE = 'Open it in ParaView and save it again as a .vtp file (File > Save Data), binary, with zlib compression or none.';

/**
 * @typedef {'points' | 'connectivity' | 'offsets' | 'cfd'} Role
 * @typedef {{ role: Role | null, type: string, components: number, format: 'binary' | 'appended' | 'ascii', offset: number }} ArrayRef
 */

/**
 * @param {Blob & { name?: string }} file
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<import('./result.js').ParsedSurface>}
 */
export async function readSurface(file, onProgress = () => {}) {
  if (file.size === 0) throw new Error('The file is empty. Choose the surface file again; it may not have finished copying.');
  const source = new SliceReader(file);
  const document = await readDocument(source, onProgress);
  const { piece, arrays, values } = document;

  for (const role of ['points', 'connectivity', 'offsets']) {
    if (!arrays[role]) throw new Error(`The file has no polygon ${role} array, so its surface cannot be rebuilt. ${RESAVE}`);
  }
  for (const role of ['points', 'connectivity', 'offsets', 'cfd']) {
    if (arrays[role]?.format === 'appended') {
      values[role] = await readAppended(source, document, arrays[role], document.progress(role));
    }
  }
  onProgress(1);
  return assembleSurface(file.name ?? '', piece, {
    points: toFloat32(values.points, arrays.points),
    connectivity: toIndices(values.connectivity, arrays.connectivity),
    ends: toIndices(values.offsets, arrays.offsets),
    cfd: values.cfd ? toFloat32(values.cfd, arrays.cfd) : null,
  });
}

/** Check the arrays against each other and against the piece's counts, and build the contract's shape. */
function assembleSurface(name, piece, { points, connectivity, ends, cfd }) {
  const count = (n) => n.toLocaleString('en-US');
  if (points.length !== 3 * piece.points) {
    throw new Error(`The file declares ${count(piece.points)} points but holds ${count(Math.floor(points.length / 3))}: it is damaged. ${RESAVE}`);
  }
  if (ends.length !== piece.polys) {
    throw new Error(`The file declares ${count(piece.polys)} polygons but describes ${count(ends.length)}: it is damaged. ${RESAVE}`);
  }
  const offsets = new Uint32Array(ends.length + 1);
  offsets.set(ends, 1);
  let previous = 0;
  for (let c = 1; c < offsets.length; c++) {
    if (offsets[c] < previous + 3) {
      throw new Error(`Polygon ${count(c - 1)} of the file has fewer than three corners: the file is damaged. ${RESAVE}`);
    }
    previous = offsets[c];
  }
  // A writer may store more connectivity than the polygons use (DrivAerML's headers overstate it); the offsets decide.
  if (previous > connectivity.length) {
    throw new Error(`The file's polygons list ${count(previous)} corners but it stores ${count(connectivity.length)}: it is damaged. ${RESAVE}`);
  }
  const corners = previous === connectivity.length ? connectivity : connectivity.subarray(0, previous);
  let largest = 0;
  for (let i = 0; i < corners.length; i++) if (corners[i] > largest) largest = corners[i];
  if (largest >= piece.points) {
    throw new Error(`The file's polygons refer to points it does not contain: it is damaged. ${RESAVE}`);
  }
  if (cfd && cfd.length !== piece.polys) {
    throw new Error(`The file's ${CFD_ARRAY} array has ${count(cfd.length)} values for ${count(piece.polys)} polygons: it is damaged. ${RESAVE}`);
  }
  return { name, points, connectivity: corners, offsets, cfd };
}

// ---------------------------------------------------------------------------------------------
// The XML structure

/**
 * Walk the XML from the top. Needed inline arrays are decoded on the way; appended ones are recorded
 * for reading afterwards. Stops at the appended data block, or as soon as every needed inline array
 * is in.
 */
async function readDocument(source, onProgress) {
  const head = latin1.decode(await source.bytes(0, 64)).trimStart();
  if (head.startsWith('# vtk DataFile')) {
    throw new Error(`This is a legacy VTK file; the tool reads VTK XML PolyData. ${RESAVE}`);
  }
  if (!head.startsWith('<')) {
    throw new Error('This is not a VTK XML file. The tool reads surfaces saved as VTK XML PolyData (.vtp), such as DrivAerML\'s boundary files.');
  }

  const document = {
    headerBytes: 4,
    compressed: false,
    appended: null,
    piece: null,
    /** @type {Partial<Record<Role, ArrayRef>>} */
    arrays: {},
    /** @type {Partial<Record<Role, Uint8Array | Float64Array>>} */
    values: {},
    /** Progress callback for one array: its share of the reading, by its expected size. */
    progress: () => () => {},
  };
  let section = null;
  let sawVtkFile = false;
  let position = 0;

  for (;;) {
    const open = await source.indexOf(0x3c, position);
    if (open < 0) break;
    const close = await source.indexOf(0x3e, open);
    if (close < 0) throw truncated(source);
    const text = latin1.decode(await source.bytes(open + 1, close));
    position = close + 1;
    if (text.startsWith('?') || text.startsWith('!')) continue;
    const name = /^\/?\w+/.exec(text)?.[0];
    const attributes = parseAttributes(text);

    switch (name) {
      case 'VTKFile':
        sawVtkFile = true;
        readFileAttributes(document, attributes);
        break;
      case 'Piece':
        if (document.piece) {
          throw new Error('The file holds several pieces; the tool reads one surface. In ParaView, apply Merge Blocks, then Extract Surface, and save it as .vtp.');
        }
        document.piece = readPiece(attributes);
        document.progress = progressByRole(document.piece, onProgress);
        break;
      case 'Points': section = 'points'; break;
      case 'Polys': section = 'polys'; break;
      case 'CellData': section = 'cells'; break;
      case 'PointData':
      case 'FieldData':
      case 'Verts':
      case 'Lines':
      case 'Strips':
      case '/Points':
      case '/Polys':
      case '/CellData':
      case '/PointData':
      case '/FieldData':
      case '/Verts':
      case '/Lines':
      case '/Strips':
        section = null;
        break;
      case 'DataArray': {
        const array = readArrayAttributes(section, attributes);
        if (array.role && !document.piece) throw new Error(`The file's arrays come before its Piece. ${RESAVE}`);
        if (array.role) document.arrays[array.role] = array;
        if (array.format === 'appended' || text.endsWith('/')) break;
        if (array.role) {
          const read = await readInline(source, document, array, position);
          document.values[array.role] = read.values;
          position = read.end;
        } else {
          position = await skipInline(source, document, array, position);
        }
        break;
      }
      case 'AppendedData': {
        const underscore = await source.indexOf(0x5f, position);
        if (underscore < 0) throw truncated(source);
        const encoding = attributes.encoding ?? 'base64';
        if (encoding !== 'raw' && encoding !== 'base64') {
          throw new Error(`The file's appended data is encoded as "${encoding}", which the tool does not read. ${RESAVE}`);
        }
        document.appended = { encoding, start: underscore + 1 };
        return checkDocument(document, sawVtkFile, source);
      }
      default:
        break;
    }
  }
  return checkDocument(document, sawVtkFile, source);
}

function readFileAttributes(document, attributes) {
  if (attributes.type !== 'PolyData') {
    throw new Error(`This is a VTK ${attributes.type ?? 'unknown'} file, not a surface (PolyData). In ParaView, apply Extract Surface and save the result as .vtp.`);
  }
  if (attributes.byte_order === 'BigEndian') {
    throw new Error(`The file stores its numbers big-endian, which the tool does not read. ${RESAVE}`);
  }
  if (attributes.header_type) {
    if (attributes.header_type !== 'UInt32' && attributes.header_type !== 'UInt64') {
      throw new Error(`The file uses ${attributes.header_type} block headers, which the tool does not read. ${RESAVE}`);
    }
    document.headerBytes = SCALAR_BYTES[attributes.header_type];
  }
  if (attributes.compressor) {
    if (attributes.compressor !== 'vtkZLibDataCompressor') {
      const kind = attributes.compressor.replace(/^vtk|DataCompressor$/g, '');
      throw new Error(`The file is compressed with ${kind}, which browsers cannot decompress. ${RESAVE}`);
    }
    document.compressed = true;
  }
}

function readPiece(attributes) {
  const piece = {
    points: Number(attributes.NumberOfPoints ?? 0),
    polys: Number(attributes.NumberOfPolys ?? 0),
  };
  const strips = Number(attributes.NumberOfStrips ?? 0);
  if (piece.polys === 0) {
    const lines = Number(attributes.NumberOfLines ?? 0);
    const verts = Number(attributes.NumberOfVerts ?? 0);
    const kind = strips > 0 ? 'triangle strips' : lines > 0 ? 'lines' : verts > 0 ? 'points' : 'nothing';
    throw new Error(`The file holds ${kind} but no polygons, so there is no surface to analyse. Export the car's surface as polygons (ParaView: Extract Surface, then Triangulate) and save it as .vtp.`);
  }
  if (strips > 0) {
    throw new Error('Part of the surface is stored as triangle strips, which the tool does not read. In ParaView, apply Triangulate and save it again as .vtp.');
  }
  return piece;
}

function readArrayAttributes(section, attributes) {
  const format = attributes.format ?? 'ascii';
  if (format !== 'binary' && format !== 'appended' && format !== 'ascii') {
    throw new Error(`The file stores an array in the format "${format}", which the tool does not read. ${RESAVE}`);
  }
  const type = attributes.type;
  if (!(type in SCALAR_BYTES)) throw new Error(`The file stores an array as ${type}, which the tool does not read. ${RESAVE}`);
  const array = {
    role: arrayRole(section, attributes),
    type,
    components: Number(attributes.NumberOfComponents ?? 1),
    format,
    offset: Number(attributes.offset ?? 0),
  };
  const integer = type.includes('Int');
  switch (array.role) {
    case 'points':
      if (integer || array.components !== 3) {
        throw new Error(`The file's points are not 3D coordinates (${array.components} ${type} per point). ${RESAVE}`);
      }
      break;
    case 'connectivity':
    case 'offsets':
      if (!integer) throw new Error(`The file stores polygon ${array.role} as ${type} instead of integers. ${RESAVE}`);
      break;
    case 'cfd':
      if (integer || array.components !== 1) throw new Error(`The file's ${CFD_ARRAY} array is not one number per polygon. ${RESAVE}`);
      break;
    case null:
      break;
    default:
      throw new Error(`Unknown array role ${array.role}`);
  }
  if (format === 'appended' && !Number.isFinite(array.offset)) {
    throw new Error(`The file's ${attributes.Name ?? 'unnamed'} array has no place in the appended data. ${RESAVE}`);
  }
  return array;
}

function arrayRole(section, attributes) {
  switch (section) {
    case 'points': return 'points';
    case 'polys':
      if (attributes.Name === 'connectivity') return 'connectivity';
      if (attributes.Name === 'offsets') return 'offsets';
      return null;
    case 'cells': return attributes.Name === CFD_ARRAY ? 'cfd' : null;
    case null: return null;
    default: throw new Error(`Unknown section ${section}`);
  }
}

function checkDocument(document, sawVtkFile, source) {
  if (!sawVtkFile) {
    throw new Error('This is not a VTK XML file. The tool reads surfaces saved as VTK XML PolyData (.vtp), such as DrivAerML\'s boundary files.');
  }
  if (!document.piece) {
    if (source.reachedEnd) throw truncated(source);
    throw new Error(`The file holds no surface piece. ${RESAVE}`);
  }
  if (Object.values(document.arrays).some((array) => array.format === 'appended') && !document.appended) {
    throw truncated(source);
  }
  return document;
}

/** Spread reading progress over the needed arrays by their expected value counts. */
function progressByRole(piece, onProgress) {
  const expected = { points: 3 * piece.points, connectivity: 4 * piece.polys, offsets: piece.polys, cfd: piece.polys };
  const total = expected.points + expected.connectivity + expected.offsets + expected.cfd;
  const done = { points: 0, connectivity: 0, offsets: 0, cfd: 0 };
  let reported = 0;
  return (role) => (fraction) => {
    done[role] = Math.min(1, fraction) * expected[role];
    const overall = Math.min(0.99, (done.points + done.connectivity + done.offsets + done.cfd) / total);
    if (overall > reported) {
      reported = overall;
      onProgress(overall);
    }
  };
}

function parseAttributes(text) {
  const attributes = {};
  for (const match of text.matchAll(/([\w:]+)\s*=\s*(['"])(.*?)\2/g)) attributes[match[1]] = match[3];
  return attributes;
}

function truncated(source) {
  return new Error(`The file ends early, at ${(source.size / 1e6).toFixed(1)} MB: it was cut off while copying or saving. Copy or export it again.`);
}

// ---------------------------------------------------------------------------------------------
// Inline arrays

/**
 * Decode an inline array's values, returning them and the position of the tag after its content.
 * Base64 content runs to the next '<', which base64 never contains; the end is found from the data
 * itself rather than the header, because some writers' headers overstate the size.
 */
async function readInline(source, document, array, position) {
  const onProgress = document.progress(array.role);
  const expectedBytes = expectedValues(array, document.piece) * SCALAR_BYTES[array.type];
  if (array.format === 'ascii') return readAscii(source, position, (bytes) => onProgress(bytes / (expectedBytes * 3)));
  const start = await skipWhitespace(source, position);
  const word = document.headerBytes;
  if (document.compressed) {
    const header = await readBlockTable(source, word, start);
    const compressed = new Uint8Array(header.table.compressedBytes);
    const end = await decodeBase64Into(source, header.dataStart, compressed, (f) => onProgress(0.5 * f));
    const values = await inflateBlocks(compressed, header.table, (f) => onProgress(0.5 + 0.5 * f));
    return { values, end: await expectTag(source, end) };
  }
  // A plain array's header and values are one base64 stream, unless the header block is padded.
  const separate = latin1.decode(await source.bytes(start, start + base64Length(word))).endsWith('=');
  const dataStart = separate ? start + base64Length(word) : start;
  const { parts, end } = await decodeBase64ToTag(source, dataStart, (bytes) => onProgress(bytes / expectedBytes));
  return { values: concat(parts, separate ? 0 : word), end };
}

/** Position of the tag after an inline array the engine does not use, found without decoding it. */
async function skipInline(source, document, array, position) {
  if (array.format === 'ascii') return expectTag(source, position);
  const start = await skipWhitespace(source, position);
  const word = document.headerBytes;
  let end;
  if (document.compressed) {
    const header = await readBlockTable(source, word, start);
    end = header.dataStart + base64Length(header.table.compressedBytes);
  } else {
    const separate = latin1.decode(await source.bytes(start, start + base64Length(word))).endsWith('=');
    const header = decodeBase64(latin1.decode(await source.bytes(start, start + base64Length(word + 2))));
    const byteCount = readWord(header, 0, word);
    end = separate ? start + base64Length(word) + base64Length(byteCount) : start + base64Length(word + byteCount);
  }
  // Trust the header's jump only if it lands just after base64 and on the closing tag; otherwise walk
  // the data to find it.
  if (end > start && end < source.size && isBase64Character((await source.bytes(end - 1, end))[0])) {
    const landing = await skipWhitespace(source, end);
    if ((await source.bytes(landing, landing + 1))[0] === 0x3c) return landing;
  }
  return expectTag(source, start);
}

function isBase64Character(byte) {
  return (byte >= 0x41 && byte <= 0x5a) || (byte >= 0x61 && byte <= 0x7a) || (byte >= 0x30 && byte <= 0x39)
    || byte === 0x2b || byte === 0x2f || byte === 0x3d;
}

async function expectTag(source, position) {
  const tag = await source.indexOf(0x3c, position);
  if (tag < 0) throw truncated(source);
  return tag;
}

async function readAscii(source, position, onProgress) {
  const end = await expectTag(source, position);
  const numbers = [];
  let carry = '';
  for (let at = position; at < end; at += SLICE_BYTES) {
    const next = Math.min(end, at + SLICE_BYTES);
    const tokens = (carry + latin1.decode(await source.bytes(at, next))).split(/\s+/);
    carry = next < end ? tokens.pop() : '';
    for (const token of tokens) if (token) numbers.push(Number(token));
    onProgress(next - position);
  }
  const values = Float64Array.from(numbers);
  if (values.some(Number.isNaN)) throw new Error(`An array in the file holds text that is not a number: it is damaged. ${RESAVE}`);
  return { values, end };
}

/** Decode base64 from `start` up to the next '<', slice by slice. Returns the decoded slices. */
async function decodeBase64ToTag(source, start, onProgress) {
  const parts = [];
  let decoded = 0;
  let carry = '';
  let position = start;
  for (;;) {
    const bytes = await source.bytes(position, position + SLICE_BYTES);
    if (bytes.length === 0) throw truncated(source);
    const tag = bytes.indexOf(0x3c);
    const content = tag < 0 ? bytes : bytes.subarray(0, tag);
    position += content.length;
    const text = carry + latin1.decode(content);
    const out = new Uint8Array(Math.ceil(text.length / 4) * 3);
    const { read, written } = setFromBase64(out, text, tag < 0 ? 'stop-before-partial' : 'loose');
    parts.push(out.subarray(0, written));
    decoded += written;
    onProgress(decoded);
    if (tag >= 0) return { parts, end: position };
    carry = text.slice(read);
  }
}

/** Decode base64 from `start` until `target` is full. Returns the position after the last character used. */
async function decodeBase64Into(source, start, target, onProgress) {
  let position = start;
  let written = 0;
  let carry = '';
  while (written < target.length) {
    const want = Math.min(SLICE_BYTES, base64Length(target.length - written) - carry.length);
    const bytes = await source.bytes(position, position + want);
    if (bytes.length === 0) throw truncated(source);
    position += bytes.length;
    const text = carry + latin1.decode(bytes);
    const result = setFromBase64(target.subarray(written), text, 'stop-before-partial');
    written += result.written;
    carry = text.slice(result.read);
    onProgress(written / target.length);
  }
  return position - carry.length;
}

function setFromBase64(target, text, lastChunkHandling) {
  try {
    return target.setFromBase64(text, { lastChunkHandling });
  } catch {
    throw new Error(`The file's binary data is not valid base64: it is damaged. ${RESAVE}`);
  }
}

function decodeBase64(text) {
  try {
    return Uint8Array.fromBase64(text, { lastChunkHandling: 'stop-before-partial' });
  } catch {
    throw new Error(`The file's binary data is not valid base64: it is damaged. ${RESAVE}`);
  }
}

function base64Length(bytes) {
  return 4 * Math.ceil(bytes / 3);
}

// ---------------------------------------------------------------------------------------------
// Appended arrays

async function readAppended(source, document, array, onProgress) {
  const word = document.headerBytes;
  const start = document.appended.start + array.offset;
  if (document.appended.encoding === 'raw') {
    if (document.compressed) {
      const blocks = readWord(await source.bytesExactly(start, start + word), 0, word);
      const tableBytes = (3 + blocks) * word;
      const table = blockTable(await source.bytesExactly(start, start + tableBytes), word);
      const compressed = await source.copy(start + tableBytes, start + tableBytes + table.compressedBytes, (f) => onProgress(0.5 * f));
      return inflateBlocks(compressed, table, (f) => onProgress(0.5 + 0.5 * f));
    }
    const byteCount = readWord(await source.bytesExactly(start, start + word), 0, word);
    return source.copy(start + word, start + word + byteCount, onProgress);
  }
  if (document.compressed) {
    const header = await readBlockTable(source, word, start);
    const compressed = new Uint8Array(header.table.compressedBytes);
    await decodeBase64Into(source, header.dataStart, compressed, (f) => onProgress(0.5 * f));
    return inflateBlocks(compressed, header.table, (f) => onProgress(0.5 + 0.5 * f));
  }
  const separate = latin1.decode(await source.bytesExactly(start, start + base64Length(word))).endsWith('=');
  const header = decodeBase64(latin1.decode(await source.bytesExactly(start, start + base64Length(word + 2))));
  const byteCount = readWord(header, 0, word);
  if (separate) {
    const values = new Uint8Array(byteCount);
    await decodeBase64Into(source, start + base64Length(word), values, onProgress);
    return values;
  }
  const stream = new Uint8Array(word + byteCount);
  await decodeBase64Into(source, start, stream, onProgress);
  return stream.subarray(word);
}

// ---------------------------------------------------------------------------------------------
// Compressed data

/** A base64 compressed array's header: its own padded block, holding the block table. */
async function readBlockTable(source, word, start) {
  const leading = decodeBase64(latin1.decode(await source.bytesExactly(start, start + base64Length(3 * word))));
  const blocks = readWord(leading, 0, word);
  const headerChars = base64Length((3 + blocks) * word);
  const header = decodeBase64(latin1.decode(await source.bytesExactly(start, start + headerChars)));
  return { table: blockTable(header, word), dataStart: start + headerChars };
}

/** VTK's compressed-array header: block count, block size, last block's size, then each block's compressed size. */
function blockTable(header, word) {
  const blocks = readWord(header, 0, word);
  const blockSize = readWord(header, 1, word);
  const lastSize = readWord(header, 2, word) || blockSize;
  const compressed = [];
  let compressedBytes = 0;
  for (let b = 0; b < blocks; b++) {
    const size = readWord(header, 3 + b, word);
    compressed.push(size);
    compressedBytes += size;
  }
  const byteCount = blocks === 0 ? 0 : (blocks - 1) * blockSize + lastSize;
  return { blocks, blockSize, lastSize, compressed, compressedBytes, byteCount };
}

/** Inflate VTK's zlib blocks, each its own zlib stream, into one array, several at a time. */
async function inflateBlocks(compressed, table, onProgress) {
  const out = new Uint8Array(table.byteCount);
  const starts = [];
  let at = 0;
  for (const size of table.compressed) {
    starts.push(at);
    at += size;
  }
  const damaged = (b) => new Error(`The file's compressed data is damaged (block ${b + 1} of ${table.blocks}). ${RESAVE}`);
  const inflateOne = async (b) => {
    const expected = b === table.blocks - 1 ? table.lastSize : table.blockSize;
    const input = compressed.subarray(starts[b], starts[b] + table.compressed[b]);
    let written = 0;
    try {
      for await (const chunk of new Blob([input]).stream().pipeThrough(new DecompressionStream('deflate'))) {
        if (written + chunk.length > expected) throw damaged(b);
        out.set(chunk, b * table.blockSize + written);
        written += chunk.length;
      }
    } catch {
      throw damaged(b);
    }
    if (written !== expected) throw damaged(b);
  };
  let next = 0;
  let finished = 0;
  const lane = async () => {
    while (next < table.blocks) {
      await inflateOne(next++);
      finished++;
      onProgress(finished / table.blocks);
    }
  };
  await Promise.all(Array.from({ length: 16 }, lane));
  return out;
}

function readWord(bytes, index, word) {
  const view = new DataView(bytes.buffer, bytes.byteOffset + index * word, word);
  if (word === 4) return view.getUint32(0, true);
  const value = view.getBigUint64(0, true);
  if (value > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error(`The file's block header is not a size: it is damaged. ${RESAVE}`);
  return Number(value);
}

// ---------------------------------------------------------------------------------------------
// Typed views

function expectedValues(array, piece) {
  switch (array.role) {
    case 'points': return 3 * piece.points;
    case 'connectivity': return 4 * piece.polys;
    case 'offsets':
    case 'cfd': return piece.polys;
    default: throw new Error(`Unknown array role ${array.role}`);
  }
}

/** @returns {Float32Array} */
function toFloat32(values, array) {
  if (values instanceof Float64Array) return new Float32Array(values);
  switch (array.type) {
    case 'Float32': return view(values, Float32Array);
    case 'Float64': return new Float32Array(view(values, Float64Array));
    default: throw new Error(`Unknown float type ${array.type}`);
  }
}

/** Indices as Uint32, rejecting negative values and values past 2³². @returns {Uint32Array} */
function toIndices(values, array) {
  const bad = () => new Error(`The file's polygon ${array.role} hold negative or oversized values: it is damaged. ${RESAVE}`);
  if (values instanceof Float64Array) {
    const out = new Uint32Array(values.length);
    for (let i = 0; i < out.length; i++) {
      const value = values[i];
      if (!(Number.isInteger(value) && value >= 0 && value < 2 ** 32)) throw bad();
      out[i] = value;
    }
    return out;
  }
  switch (array.type) {
    case 'UInt32': return view(values, Uint32Array);
    case 'Int32': {
      const signed = view(values, Int32Array);
      for (let i = 0; i < signed.length; i++) if (signed[i] < 0) throw bad();
      return new Uint32Array(signed.buffer, signed.byteOffset, signed.length);
    }
    case 'Int64':
    case 'UInt64': {
      const words = view(values, Uint32Array);
      const out = new Uint32Array(words.length / 2);
      for (let i = 0; i < out.length; i++) {
        if (words[2 * i + 1] !== 0) throw bad();
        out[i] = words[2 * i];
      }
      return out;
    }
    default:
      throw new Error(`The file stores polygon ${array.role} as ${array.type}, which the tool does not read. ${RESAVE}`);
  }
}

function view(bytes, Type) {
  if (bytes.byteLength % Type.BYTES_PER_ELEMENT !== 0) {
    throw new Error(`An array in the file has a size that does not fit its type: it is damaged. ${RESAVE}`);
  }
  const aligned = bytes.byteOffset % Type.BYTES_PER_ELEMENT === 0 ? bytes : bytes.slice();
  return new Type(aligned.buffer, aligned.byteOffset, aligned.byteLength / Type.BYTES_PER_ELEMENT);
}

function concat(parts, skip) {
  const total = parts.reduce((sum, part) => sum + part.length, 0) - skip;
  if (total < 0) throw new Error(`An array in the file is shorter than its header: it is damaged. ${RESAVE}`);
  const out = new Uint8Array(total);
  let at = -skip;
  for (const part of parts) {
    if (at >= 0) out.set(part, at);
    else if (at + part.length > 0) out.set(part.subarray(-at), 0);
    at += part.length;
  }
  return out;
}

async function skipWhitespace(source, position) {
  for (;;) {
    const bytes = await source.bytes(position, position + 256);
    if (bytes.length === 0) throw truncated(source);
    for (let i = 0; i < bytes.length; i++) {
      const b = bytes[i];
      if (b !== 0x20 && b !== 0x0a && b !== 0x0d && b !== 0x09) return position + i;
    }
    position += bytes.length;
  }
}

// ---------------------------------------------------------------------------------------------
// Writing the file back with cell arrays added

/** Uncompressed size of the zlib blocks of an added array. */
const DEFLATE_BLOCK_BYTES = 1 << 20;

/**
 * The file with Float32 cell arrays added, composed from slices of it rather than a copy: the new
 * arrays go into its appended data (a new raw block when it has none), stored as its other binary
 * data is (header width, zlib), and only their tags are added to its cell data. A cell array of the
 * same name already in the file is replaced; every other byte is the file's own.
 * @param {Blob} file  a file `readSurface` has read
 * @param {Record<string, Float32Array>} arrays  by name, one value per polygon in the file's order
 * @param {(fraction: number) => void} [onProgress]
 * @returns {Promise<Blob>}
 */
export async function addCellArrays(file, arrays, onProgress = () => {}) {
  const source = new SliceReader(file);
  const names = Object.keys(arrays);
  const layout = await locateCellData(source, names, (fraction) => onProgress(0.5 * fraction));
  const encoding = layout.appended?.encoding ?? 'raw';
  const end = await lastTagIn(file, layout.appended ? '</AppendedData>' : '</VTKFile>');

  const stored = [];
  for (const [k, values] of Object.values(arrays).entries()) {
    const { header, body } = await binaryArray(values, layout.document);
    const parts = encoding === 'raw' ? [header, ...body] : [header.toBase64(), concat(body, 0).toBase64()];
    stored.push({ parts, length: parts.reduce((sum, part) => sum + part.length, 0) });
    onProgress(0.5 + 0.5 * (k + 1) / names.length);
  }
  // Offsets count from just after the appended data's '_', in stored bytes or base64 characters.
  let offset = layout.appended ? end - layout.appended.start : 0;
  const tags = names.map((name, k) => {
    const tag = `<DataArray type="Float32" Name="${name}" format="appended" offset="${offset}"/>\n`;
    offset += stored[k].length;
    return tag;
  });
  const cellData = layout.cellData.wrap ? `<CellData>\n${tags.join('')}</CellData>\n` : tags.join('');
  const data = stored.flatMap((array) => array.parts);
  const edits = [
    ...layout.replaced.map(([from, to]) => ({ from, to, parts: [] })),
    { from: layout.cellData.from, to: layout.cellData.to, parts: [cellData] },
    { from: end, to: end, parts: layout.appended ? data : ['<AppendedData encoding="raw">\n_', ...data, '\n</AppendedData>\n'] },
  ].sort((a, b) => a.from - b.from);

  const parts = [];
  let cursor = 0;
  for (const edit of edits) {
    parts.push(file.slice(cursor, edit.from), ...edit.parts);
    cursor = edit.to;
  }
  parts.push(file.slice(cursor));
  return new Blob(parts);
}

/**
 * Walk the XML to where new cell array tags go: before `</CellData>`, or in place of an empty
 * `<CellData/>`, or at the start of the piece when there is no cell data. Records the cell arrays
 * named `names` (to be replaced), the file's binary settings and its appended data block.
 */
async function locateCellData(source, names, onProgress) {
  const document = { headerBytes: 4, compressed: false };
  /** @type {number[][]} [from, to) of each cell array to replace */
  const replaced = [];
  let cellData = null;
  let pieceStart = null;
  let inCellData = false;
  let replacing = null;
  let appended = null;
  let position = 0;

  walk: for (;;) {
    const open = await source.indexOf(0x3c, position);
    if (open < 0) break;
    const close = await source.indexOf(0x3e, open);
    if (close < 0) throw truncated(source);
    const text = latin1.decode(await source.bytes(open + 1, close));
    position = close + 1;
    onProgress(position / source.size);
    if (text.startsWith('?') || text.startsWith('!')) continue;
    const name = /^\/?\w+/.exec(text)?.[0];
    const attributes = parseAttributes(text);
    const selfClosing = text.endsWith('/');

    switch (name) {
      case 'VTKFile':
        readFileAttributes(document, attributes);
        break;
      case 'Piece':
        pieceStart = position;
        break;
      case 'CellData':
        if (selfClosing) cellData = { from: open, to: position, wrap: true };
        else inCellData = true;
        break;
      case '/CellData':
        cellData = { from: open, to: open, wrap: false };
        inCellData = false;
        break;
      case 'DataArray': {
        const replace = inCellData && names.includes(attributes.Name);
        if (replace && selfClosing) replaced.push([open, position]);
        else if (replace) replacing = open;
        const format = attributes.format ?? 'ascii';
        if (!selfClosing && format !== 'appended') position = await skipInline(source, document, { format }, position);
        break;
      }
      case '/DataArray':
        if (replacing !== null) replaced.push([replacing, position]);
        replacing = null;
        break;
      case 'AppendedData': {
        const underscore = await source.indexOf(0x5f, position);
        if (underscore < 0) throw truncated(source);
        appended = { encoding: attributes.encoding ?? 'base64', start: underscore + 1 };
        break walk;
      }
      default:
        break;
    }
  }
  if (pieceStart === null) throw truncated(source);
  return { document, replaced, appended, cellData: cellData ?? { from: pieceStart, to: pieceStart, wrap: true } };
}

/** Position of the last `tag` in the file, which ends with it and its parents' closing tags. */
async function lastTagIn(file, tag) {
  const from = Math.max(0, file.size - 4096);
  const at = latin1.decode(new Uint8Array(await file.slice(from).arrayBuffer())).lastIndexOf(tag);
  if (at < 0) throw new Error(`The file has no ${tag} at its end: it was cut off or changed since it was read. Analyse it again.`);
  return from + at;
}

/** One array as the file stores binary data: a size header and the values, or VTK's table and zlib blocks. */
async function binaryArray(values, { headerBytes, compressed }) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  if (!compressed) return { header: words(headerBytes, [bytes.length]), body: [bytes] };
  const blocks = [];
  for (let at = 0; at < bytes.length; at += DEFLATE_BLOCK_BYTES) blocks.push(bytes.subarray(at, at + DEFLATE_BLOCK_BYTES));
  const body = await Promise.all(blocks.map(async (block) => {
    const stream = new Blob([block]).stream().pipeThrough(new CompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }));
  const table = [blocks.length, DEFLATE_BLOCK_BYTES, bytes.length % DEFLATE_BLOCK_BYTES, ...body.map((block) => block.length)];
  return { header: words(headerBytes, table), body };
}

/** Little-endian unsigned words of `word` bytes. */
function words(word, values) {
  const bytes = new Uint8Array(word * values.length);
  const view = new DataView(bytes.buffer);
  values.forEach((value, i) => {
    if (word === 4) view.setUint32(4 * i, value, true);
    else view.setBigUint64(8 * i, BigInt(value), true);
  });
  return bytes;
}

// ---------------------------------------------------------------------------------------------
// Reading the file in slices

/**
 * Forward reads over a Blob through one window of bytes, reading one slice ahead so the next slice
 * is on its way while the current one is decoded.
 */
class SliceReader {
  /** @param {Blob} file */
  constructor(file) {
    this.file = file;
    this.size = file.size;
    this.windowStart = 0;
    this.window = new Uint8Array(0);
    this.ahead = null;
    this.reachedEnd = false;
  }

  /** Bytes [from, to), fewer at the end of the file. The view is valid until the next call. */
  async bytes(from, to) {
    to = Math.min(to, this.size);
    if (to <= from) return new Uint8Array(0);
    const windowEnd = this.windowStart + this.window.length;
    if (from >= this.windowStart && to <= windowEnd) {
      return this.window.subarray(from - this.windowStart, to - this.windowStart);
    }
    if (from < this.windowStart || from > windowEnd) {
      this.windowStart = from;
      this.window = new Uint8Array(0);
    }
    const parts = [this.window.subarray(from - this.windowStart)];
    let have = this.windowStart + this.window.length;
    while (have < to) {
      const slice = await this.slice(have);
      parts.push(slice);
      have += slice.length;
    }
    if (have >= this.size) this.reachedEnd = true;
    this.windowStart = from;
    this.window = parts.length === 1 ? parts[0] : concat(parts, 0);
    return this.window.subarray(0, to - from);
  }

  /** Exactly bytes [from, to), or a truncation error. */
  async bytesExactly(from, to) {
    const bytes = await this.bytes(from, to);
    if (bytes.length < to - from) throw truncated(this);
    return bytes;
  }

  /** A fresh array of bytes [from, to), read straight from the file. */
  async copy(from, to, onProgress) {
    if (to > this.size) throw truncated(this);
    const out = new Uint8Array(to - from);
    for (let at = from; at < to; at += SLICE_BYTES) {
      const end = Math.min(to, at + SLICE_BYTES);
      out.set(new Uint8Array(await this.file.slice(at, end).arrayBuffer()), at - from);
      onProgress((end - from) / (to - from));
    }
    return out;
  }

  /** Index of the first `byte` at or after `from`, or -1 when the file ends first. */
  async indexOf(byte, from) {
    for (let position = from; ;) {
      const bytes = await this.bytes(position, position + 4096);
      if (bytes.length === 0) return -1;
      const found = bytes.indexOf(byte);
      if (found >= 0) return position + found;
      position += bytes.length;
    }
  }

  async slice(at) {
    const read = this.ahead?.at === at ? this.ahead.read : this.read(at);
    const slice = await read;
    const next = at + slice.length;
    this.ahead = next < this.size ? { at: next, read: this.read(next) } : null;
    return slice;
  }

  async read(at) {
    return new Uint8Array(await this.file.slice(at, Math.min(this.size, at + SLICE_BYTES)).arrayBuffer());
  }
}
