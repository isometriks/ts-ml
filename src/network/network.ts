import Layer from "./layer.ts";
import InputLayer from "./input-layer.ts";
import Synapse from "../neuron/synapse.ts";

export default class Network {
  #inputLayer: InputLayer
  #outputLayer: Layer
  #hiddenLayers: Layer[] = []

  constructor(inputs: number, outputs: number, hiddenLayers: number[] = []) {
    this.#inputLayer = new InputLayer(inputs, undefined, false)
    this.#outputLayer = new Layer(outputs, undefined, false)

    let lastLayer: Layer | null = null;

    for (const hiddenNodes of  hiddenLayers) {
      const hiddenLayer = new Layer(hiddenNodes, undefined, false)
      this.#hiddenLayers.push(hiddenLayer)

      if (lastLayer) {
        lastLayer.attachLayer(hiddenLayer)
      } else {
        this.#inputLayer.attachLayer(hiddenLayer)
      }

      lastLayer = hiddenLayer
    }

    if (lastLayer) {
      lastLayer.attachLayer(this.#outputLayer)
    } else {
      // No hidden layers just connect input to output
      this.#inputLayer.attachLayer(this.#outputLayer)
    }
  }

  compute(inputs: number[]) {
    this.#inputLayer.setInputs(inputs)

    return this.#outputLayer.output()
  }

  train(inputs: number[], outputs: number[], learningRate: number = 0.2) {
    const computedOutput = this.compute(inputs)
    const adjustments: [Synapse, number, number][] = []
    const layers = [...this.#hiddenLayers, this.#outputLayer]

    for (const layer: Layer of layers.toReversed()) {
      layer.neurons.forEach((outputNeuron, index) => {
        const grad = layer === this.#outputLayer ?
          this.#calculateOutputGradient(outputNeuron, outputs[index] - computedOutput[index]) :
          this.#calculateHiddenGradient(outputNeuron)

        outputNeuron.sigma = grad
        outputNeuron.synapses.forEach(synapse => {
          adjustments.push([synapse, grad, synapse.neuron.output()])
        })
      })
    }

    adjustments.forEach(([synapse, grad, output]) => {
      synapse.adjust(grad * learningRate * output)
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

  get inputLayer(): Layer {
    return this.#inputLayer
  }

  get outputLayer(): Layer {
    return this.#outputLayer
  }

  get hiddenLayers(): Layer[] {
    return this.#hiddenLayers
  }
}
