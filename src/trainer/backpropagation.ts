import Network from "../network/network.ts";
import Synapse from "../neuron/synapse.ts";

type Adjustments = Map<Synapse, number>

export default class Backpropagation {
  readonly #network: Network
  readonly #learningRate: number

  constructor(network: Network, learningRate: number = 0.5) {
    this.#network = network
    this.#learningRate = learningRate
  }

  train(sample: TrainingSample) {
    const adjustments = this.#getAdjustments(sample.inputs, sample.outputs)
    this.#applyAdjustments(adjustments)
  }

  trainBatch(samples: TrainingSample[]) {
    const allAdjustments: Adjustments[] = []
    const randomizedSamples = [...samples].sort(() => .5 - Math.random())

    for (const { inputs, outputs} of randomizedSamples)  {
      allAdjustments.push(this.#getAdjustments(inputs, outputs))
    }

    const finalAdjustment: Adjustments = new Map()

    // Average together all adjustments from batch
    for (const adjustment of allAdjustments) {
      for (const [synapse, delta] of adjustment.entries()) {
        const existing = finalAdjustment.get(synapse) ?? 0
        finalAdjustment.set(synapse, existing + delta / allAdjustments.length)
      }
    }

    this.#applyAdjustments(finalAdjustment)
  }

  #getAdjustments(inputs: number[], outputs: number[]) {
    const computedOutput = this.#network.compute(inputs)
    const adjustments: Adjustments = new Map()
    const layers = [...this.#network.hiddenLayers, this.#network.outputLayer]

    for (const layer of layers.toReversed()) {
      layer.neurons.forEach((outputNeuron, index) => {
        const grad = layer === this.#network.outputLayer ?
          this.#calculateOutputGradient(outputNeuron, outputs[index] - computedOutput[index]) :
          this.#calculateHiddenGradient(outputNeuron)

        outputNeuron.sigma = grad

        for (const synapse of outputNeuron.synapses) {
          adjustments.set(synapse, grad * synapse.neuron.output() * this.#learningRate)
        }
      })
    }

    return adjustments
  }

  #applyAdjustments(adjustments: Adjustments) {
    for (const [synapse, adjustment] of adjustments.entries()) {
      synapse.adjust(adjustment)
    }
  }

  #calculateOutputGradient(neuron: ConnectableNeuronInterface, error: number) {
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
