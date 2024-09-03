import Layer from "./layer.ts";
import InputLayer from "./input-layer.ts";
import { ActivationFunction } from "../neuron/neuron.ts";

interface LayerConfiguration {
  neurons: number
  activationFunction?: ActivationFunction
}

interface HiddenLayerConfiguration extends LayerConfiguration {
  bias?: number
}

interface NetworkExport {
  shape: {
    inputs: number,
    outputs: LayerConfiguration,
    hiddenLayers: HiddenLayerConfiguration[]
  },
  layers: {
    neurons: {
      synapses: {
        weight: number
      }[]
      bias: number
    }[]
  }[]
}

export default class Network {
  #inputsConfig: number
  #outputsConfig: LayerConfiguration
  #hiddenLayersConfig: HiddenLayerConfiguration[]

  #inputLayer: InputLayer
  #outputLayer: Layer
  #hiddenLayers: Layer[] = []

  constructor(inputs: number, outputLayer: LayerConfiguration, hiddenLayers: HiddenLayerConfiguration[] = []) {
    this.#inputsConfig = inputs
    this.#outputsConfig = outputLayer
    this.#hiddenLayersConfig = hiddenLayers

    this.#inputLayer = new InputLayer(inputs)
    this.#outputLayer = new Layer(outputLayer.neurons, outputLayer.activationFunction, 0)

    let lastLayer: Layer | null = null;

    for (const layerConfiguration of  hiddenLayers) {
      const hiddenLayer = new Layer(
        layerConfiguration.neurons,
        layerConfiguration.activationFunction,
        layerConfiguration.bias
      )

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
    this.unCache()

    return this.#outputLayer.output()
  }

  unCache() {
    [...this.hiddenLayers, this.outputLayer].forEach(layer => layer.unCache())
  }

  get inputLayer(): InputLayer {
    return this.#inputLayer
  }

  get outputLayer(): Layer {
    return this.#outputLayer
  }

  get hiddenLayers(): Layer[] {
    return this.#hiddenLayers
  }

  export(): NetworkExport {
    const layers = [this.outputLayer, ...this.hiddenLayers.toReversed()].map(layer => {
      return {
        neurons: layer.neurons.map(neuron => {
          return {
            bias: neuron.bias,
            synapses: neuron.synapses.map(synapse => {
              return { weight: synapse.weight }
            })
          }
        })
      }
    });

    return {
      shape: {
        inputs: this.#inputsConfig,
        outputs: this.#outputsConfig,
        hiddenLayers: this.#hiddenLayersConfig,
      },
      layers
    }
  }

  import(exportedNetwork: NetworkExport) {
    const layers = [this.outputLayer, ...this.hiddenLayers.toReversed()]

    exportedNetwork.layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        layers[layerIndex].neurons[neuronIndex].bias = neuron.bias

        neuron.synapses.forEach(({ weight }, synapseIndex) => {
          layers[layerIndex].neurons[neuronIndex].synapses[synapseIndex].weight = weight
        })
      })
    })
  }

  static fromNetworkExport(exportedNetwork: NetworkExport) {
    const network = new this(
      exportedNetwork.shape.inputs,
      exportedNetwork.shape.outputs,
      exportedNetwork.shape.hiddenLayers
    )

    network.import(exportedNetwork)

    return network
  }
}
