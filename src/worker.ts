import { expose } from "threads/worker"
import Network from "./network/network.ts";
import Backpropagation from "./trainer/backpropagation.ts";
import mnist from 'mnist';

const formatSamples = (mnistSamples) => {
  return mnistSamples.map(({ input, output }) => {
    return { inputs: input, outputs: output }
  })
}

expose({
  trainBatch(exportedNetwork) {
    const network = Network.fromNetworkExport(exportedNetwork)
    const trainer = new Backpropagation(network, 0.2)

    console.time("trainWorker")
    const adjustments = trainer.trainBatch(formatSamples(mnist.get(100)))
    console.timeEnd("trainWorker")

    return adjustments
  }
})
