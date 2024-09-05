import Network from "../network/network.ts";
import Layer from "../network/layer.ts";

type Adjustments = number[][][]

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

    const finalAdjustment: Adjustments = []

    // Average together all adjustments from batch
    for (const adjustment of allAdjustments) {
      adjustment.forEach((neurons, layerIndex) => {
        finalAdjustment[layerIndex] ??= []

        neurons.forEach((synapses, neuronIndex) => {
          finalAdjustment[layerIndex][neuronIndex] ??= []

          synapses.forEach((delta, synapseIndex) => {
            const existing = finalAdjustment[layerIndex][neuronIndex][synapseIndex] ?? 0
            finalAdjustment[layerIndex][neuronIndex][synapseIndex] = existing + delta / allAdjustments.length
          })
        })
      })
    }

    this.#applyAdjustments(finalAdjustment)
  }

  #getAdjustments(inputs: number[], outputs: number[]) {
    const computedOutput = this.#network.compute(inputs)
    const adjustments: Adjustments = []
    const layers: Layer[] = [this.#network.outputLayer, ...this.#network.hiddenLayers.toReversed()]

    layers.forEach((layer, layerIndex) => {
      adjustments[layerIndex] = []

      layer.neurons.forEach((neuron, neuronIndex) => {
        adjustments[layerIndex][neuronIndex] = []

        const grad = layer === this.#network.outputLayer ?
          this.#calculateOutputGradient(neuron, outputs[neuronIndex] - computedOutput[neuronIndex]) :
          this.#calculateHiddenGradient(neuron)

        neuron.sigma = grad

        neuron.synapses.forEach((synapse, synapseIndex) => {
          adjustments[layerIndex][neuronIndex][synapseIndex] =  grad * synapse.neuron.output()
        })
      })
    })

    return adjustments
  }

  #applyAdjustments(adjustments: Adjustments) {
    adjustments.forEach((neurons, layerIndex) => {
      neurons.forEach((synapses, neuronIndex) => {
        synapses.forEach((adjustment, synapseIndex) => {
          this.#network.getSynapse(layerIndex, neuronIndex, synapseIndex).adjust(adjustment  * this.#learningRate)
        })
      })
    })
  }

  #calculateOutputGradient(neuron: ConnectableNeuronInterface, error: number) {
    return neuron.derivative() * error
  }

  #calculateHiddenGradient(neuron: ConnectableNeuronInterface)
  {
    return neuron.derivative() * neuron.reverseSynapseNeurons.reduce(
      (accumulator, [prevNeuron, synapse]) => {
        return accumulator + prevNeuron.sigma * synapse.weight
      },
      0
    )
  }
}
