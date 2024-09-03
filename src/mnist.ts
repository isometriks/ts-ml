import Network from "./network/network.ts";
import Backpropagation from "./trainer/backpropagation.ts";
import { Pool, spawn, Worker } from "threads"
import mnist from 'mnist';
import { ActivationFunction } from "./neuron/neuron.ts";

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div id="graph" style="margin: 0 auto; width: 100%; height: 400px;"></div>
`

const network = new Network(
  28 * 28,
  { neurons: 10 },
  [
    { neurons: 100, activationFunction: ActivationFunction.Relu },
    //{ neurons: 100, activationFunction: ActivationFunction.Sigmoid },
  ]
)

const trainer = new Backpropagation(network, 0.2)
const formatSamples = (mnistSamples) => {
  return mnistSamples.map(({ input, output }) => {
    return { inputs: input, outputs: output }
  })
}

let cycles = 0

async function main() {
  const pool = Pool(() => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), {
      type: 'module'
    })

    return spawn(worker)
  }, 1)

  for (let i=0; i<50; i++) {
    pool.queue(async worker => {
      const exportedNetwork = network.export()
      const adjustments = await worker.trainBatch(exportedNetwork)
      trainer.applyAdjustments(adjustments)

      cycles++

      document.getElementById("cycles")!.innerHTML = `<h4 style="margin: 10px 0">${cycles} Batches Run</h4>`
    })
  }

  await pool.completed()
  await pool.terminate()

  for (const { inputs, outputs } of formatSamples(mnist.get(10))) {
    const computed = network.compute(inputs)
    const expected = mnist.toNumber(outputs)
    const calculated = mnist.toNumber(computed)

    console.log("Expecting", expected, " == ", calculated, computed)
  }

  showDigit()
  document.getElementById("correctness")!.innerHTML = `Network Correctness: ${correctness().toFixed(3)}%`
}

function correctness(): number {
  let total = 0;
  let correct = 0;

  for (const { inputs, outputs } of formatSamples(mnist.set(0, 2000).test)) {
    const computed = network.compute(inputs)
    const expected = mnist.toNumber(outputs)
    const calculated = mnist.toNumber(computed)

    total++

    if (expected === calculated) {
      correct++
    }
  }

  return (100 * correct / total)
}

function showDigit() {
  const data = mnist.get(1)[0]

  const computed = network.compute(data.input)
  const expected = mnist.toNumber(data.output)
  const calculated = mnist.toNumber(computed)

  const context = (document.getElementById<HTMLCanvasElement>('digit')).getContext('2d');
  document.getElementById("output")!.innerHTML = `
    <h5 style="margin: 5px;">Digit: ${expected} - Network Calculated: ${calculated}</h5>  
  `
  mnist.draw(data.input, context); // draws a '1' mnist digit in the canvas
}

document.getElementById("next")!.addEventListener("click", showDigit)
document.getElementById("rerun")!.addEventListener("click", async () => {
  await main()
})

await main()
