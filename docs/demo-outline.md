# Demo outline

How the tool is presented to a customer: an aerodynamics team that has a geometry and no CFD run for it. The five slides in [demo-slides.pdf](demo-slides.pdf) are shown first, then the tool is run live on the customer's file.

1. **The cost.** One DrivAerML CFD run took around 40 hours on 1,536 cores. Surrogate Model gives the pressure on all 8.8 million cells of run 1 in about 12 seconds on a laptop.
2. **Their file.** The customer's `.vtp` is dropped on the page. The pressure field is on the car about five seconds later, computed in the browser; the file is not uploaded.
3. **Worked as an engineer would.** The forces are read off the result card and the lowest pressure is flown to; the section is moved along the car and read on its plot; a probe is placed.
4. **A study, handed to the assistant.** "Raise the car 20 mm and show me where the pressure changes most." The assistant makes the variant with the same operations the engineer has, predicts it in about a second, colours the car by the change and marks where it is largest.
5. **Checked.** On a car that has CFD: the CFD and the error map on the car, the CFD's line beside the prediction on the section plot, and the spread where the five networks disagree. On cars outside training the model explains 96% of the variance of the CFD pressure. Integrated drag is not yet a ranking number: its error in cross-validation is 20 counts (a count is 0.001 of a force coefficient), against 17 for guessing the average car. Lift does better: 40 counts against 61.
6. **What more runs buy.** Everything the model is told is a description of the car's shape; it learns the flow from ten CFD runs, and that carries the pressure field to a new car. Trained on 2, 4, 6 and 8 cars it scores $R^2$ `0.84`, `0.90`, `0.93` and `0.95` on cars it did not see, with drag errors of 39, 39, 23 and 19 counts. So the loop is: predict a variant, read the field, run CFD where it needs checking, retrain with the new run.
