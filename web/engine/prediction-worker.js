/**
 * One of the engine's helper workers. For the car being analysed it computes the networks' inputs at
 * batches of points, and, when there is no GPU, the networks' outputs too; it also finds, for points
 * of the full mesh, the nearest vertex of the working mesh.
 */

import { computeFeatures } from './features.js';
import { predictOnCpu } from './network.js';
import { buildPointGrid, nearestPoint } from './nearest.js';

let ensemble = null;
let context = null;
let workingGrid = null;

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'ensemble':
      ensemble = data.ensemble;
      break;
    case 'context':
      context = data.context;
      workingGrid = null;
      break;
    case 'working':
      workingGrid = buildPointGrid(data.positions);
      break;
    // The car's analysis has ended: its lookups (tens of MB in each helper) are not needed again.
    case 'release':
      context = null;
      workingGrid = null;
      break;
    case 'features': {
      // Rows of `stride` floats, as the GPU networks take them; the padding stays zero.
      const { positions, normals, count, stride } = data;
      const features = new Float32Array(count * stride);
      computeFeatures(context, positions, normals, 0, count, features, stride);
      self.postMessage({ id: data.id, result: features }, [features.buffer]);
      break;
    }
    case 'models': {
      const { positions, normals, count } = data;
      const features = new Float32Array(count * context.layout.count);
      computeFeatures(context, positions, normals, 0, count, features);
      const models = new Float32Array(ensemble.folds.length * count);
      predictOnCpu(ensemble, features, count, models);
      self.postMessage({ id: data.id, result: models }, [models.buffer]);
      break;
    }
    case 'nearest': {
      const { positions, count } = data;
      const nearest = new Uint32Array(count);
      for (let p = 0; p < count; p++) {
        nearest[p] = nearestPoint(workingGrid, positions[3 * p], positions[3 * p + 1], positions[3 * p + 2]);
      }
      self.postMessage({ id: data.id, result: nearest }, [nearest.buffer]);
      break;
    }
    default:
      throw new Error(`Unknown task ${data.type}`);
  }
};
