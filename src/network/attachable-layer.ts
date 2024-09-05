import Layer from "./layer.ts";

export default abstract class AttachableLayer {
  abstract get neurons(): ConnectableNeuronInterface[]

  attachNeuron(neuron: ConnectableNeuronInterface) {
    for (const neighbor of this.neurons) {
      neuron.addSynapse(neighbor, Math.random())
    }
  }

  attachLayer(layer: Layer) {
    for (const neuron of layer.neurons) {
      this.attachNeuron(neuron)
    }
  }
}
