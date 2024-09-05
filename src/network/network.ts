import Layer from "./layer.ts";
import InputLayer from "./input-layer.ts";

interface LayerConfiguration {
  neurons: number
  activationFunction?: FunctionInterface
}

interface HiddenLayerConfiguration extends LayerConfiguration {
  bias?: number
}

export default class Network {
  #inputLayer: InputLayer
  #outputLayer: Layer
  #hiddenLayers: Layer[] = []

  constructor(inputs: number, outputLayer: LayerConfiguration, hiddenLayers: HiddenLayerConfiguration[] = []) {
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
}
