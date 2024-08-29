import Network from "../network/network.ts";
import Synapse from "../neuron/synapse.ts";
import Layer from "../network/layer.ts";

export default class Backpropagation {
  readonly #network: Network
  readonly #learningRate: number

  constructor(network: Network, learningRate: number = 0.5) {
    this.#network = network
    this.#learningRate = learningRate
  }

  train(inputs: number[], outputs: number[]) {
    const computedOutput = this.#network.compute(inputs)
    const adjustments: [Synapse, number, number][] = []
    const layers = [...this.#network.hiddenLayers, this.#network.outputLayer]

    for (const layer: Layer of layers.toReversed()) {
      layer.neurons.forEach((outputNeuron, index) => {
        const grad = layer === this.#network.outputLayer ?
          this.#calculateOutputGradient(outputNeuron, outputs[index] - computedOutput[index]) :
          this.#calculateHiddenGradient(outputNeuron)

        outputNeuron.sigma = grad
        outputNeuron.synapses.forEach(synapse => {
          adjustments.push([synapse, grad, synapse.neuron.output()])
        })
      })
    }

    adjustments.forEach(([synapse, grad, output]) => {
      synapse.adjust(grad * this.#learningRate * output)
    })
  }

  #calculateOutputGradient(neuron: ConnectableNeuronInterface, error) {
    return neuron.derivative() * error
  }

  #calculateHiddenGradient(neuron: ConnectableNeuronInterface)
  {
    let a = 0;

    for (const [prevNeuron, synapse] of neuron.reverseSynapseNeurons) {
      a += prevNeuron.sigma * synapse.weight
    }

    return neuron.derivative() * a
  }
}
