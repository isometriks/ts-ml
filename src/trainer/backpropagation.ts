import Network from "../network/network.ts";
import Synapse from "../neuron/synapse.ts";
import Layer from "../network/layer.ts";

type Adjustments = Map<Synapse, [number, number]>

export default class Backpropagation {
  readonly #network: Network
  readonly #learningRate: number

  constructor(network: Network, learningRate: number = 0.5) {
    this.#network = network
    this.#learningRate = learningRate
  }

  train(inputs: number[], outputs: number[]) {
    const adjustments = this.#getAdjustments(inputs, outputs)
    this.#applyAdjustments(adjustments)
  }

  #getAdjustments(inputs: number[], outputs: number[]) {
    const computedOutput = this.#network.compute(inputs)
    const adjustments: Adjustments = new Map()
    const layers = [...this.#network.hiddenLayers, this.#network.outputLayer]

    for (const layer: Layer of layers.toReversed()) {
      layer.neurons.forEach((outputNeuron, index) => {
        const grad = layer === this.#network.outputLayer ?
          this.#calculateOutputGradient(outputNeuron, outputs[index] - computedOutput[index]) :
          this.#calculateHiddenGradient(outputNeuron)

        outputNeuron.sigma = grad
        outputNeuron.synapses.forEach(synapse => {
          adjustments.set(synapse, [grad, synapse.neuron.output()])
        })
      })
    }

    return adjustments
  }

  #applyAdjustments(adjustments: Adjustments) {
    for (const [synapse, [grad, output]] of adjustments.entries()) {
      synapse.adjust(grad * this.#learningRate * output)
    }
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
