//import './style.css'
import Network from "./network/network.ts";
import Renderer from "./render/renderer.ts";
import Backpropagation from "./trainer/backpropagation.ts";

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div id="graph" style="margin: 0 auto; width: 100%; height: 400px;"></div>
`

// [inputs, expected output]
const xorSamples = [
  [[0, 0], [0]],
  [[0, 1], [1]],
  [[1, 0], [1]],
  [[1, 1], [0]],
];

const network = new Network(2, 1, [3, 3])
const renderer = new Renderer(document.getElementById("graph")!)
const trainer = new Backpropagation(network, 0.3)

console.time("train")
for (let i = 0; i < 200000; i++) {
  trainer.trainBatch(xorSamples)
}
console.timeEnd("train")

network.compute([1, 0])

renderer.draw(network)

for (const [inputs, outputs] of xorSamples) {
  console.log("Inputs", inputs, "should be", outputs, " == ", network.compute(inputs))
}
