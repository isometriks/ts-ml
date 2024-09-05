import Layer from "./layer.ts";
import InputLayer from "./input-layer.ts";

interface HiddenLayerConfiguration {
  neurons: number
  activationFunction?: FunctionInterface
  bias?: number
}

export default class Network {
  #inputLayer: InputLayer
  #outputLayer: Layer
  #hiddenLayers: Layer[] = []

  constructor(inputs: number, outputs: number, hiddenLayers: HiddenLayerConfiguration[] = []) {
    this.#inputLayer = new InputLayer(inputs)
    this.#outputLayer = new Layer(outputs, undefined, 0)

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
