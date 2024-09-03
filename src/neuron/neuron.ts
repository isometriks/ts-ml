import Synapse from "./synapse.ts";
import Relu from "../function/relu.ts";
import Sigmoid from "../function/sigmoid.ts";

export enum ActivationFunction {
  Sigmoid,
  Relu,
}

const getActivationInstance = (type: ActivationFunction): FunctionInterface => {
  switch(type) {
    case ActivationFunction.Sigmoid: return Sigmoid.instance()
    case ActivationFunction.Relu: return Relu.instance()
  }
}

export default class Neuron implements ConnectableNeuronInterface {
  static neuronCount = 0

  #output: number | null = null
  #func: FunctionInterface
  #synapses: Synapse[] = []
  #reverseSynapseNeurons: [NeuronInterface, Synapse][] = []
  #sigma: number = 0
  #id: number
  #bias: number

  constructor(activationFunction: ActivationFunction = ActivationFunction.Sigmoid, bias: number = 0) {
    this.#func = getActivationInstance(activationFunction)
    this.#id = Neuron.neuronCount++
    this.#bias = Math.random() * (bias * 2) - bias
  }

  addSynapse(neuron: NeuronInterface, weight: number = 1.0) {
    const synapse = new Synapse(neuron, weight)
    this.#synapses.push(synapse)
    neuron.addReverseSynapseNeuron(this, synapse)
  }

  addReverseSynapseNeuron(neuron: NeuronInterface, synapse: Synapse) {
    this.#reverseSynapseNeurons.push([neuron, synapse])
  }

  unCache() {
    this.#output = null
  }

  get synapses() {
    return this.#synapses
  }

  get reverseSynapseNeurons() {
    return this.#reverseSynapseNeurons
  }

  get identifier() {
    return `neuron_${this.#id}`
  }

  get label() {
    return `Neuron ${this.#id}: ${this.output().toFixed(4)}`
  }

  set sigma(sigma: number) {
    this.#sigma = sigma
  }

  get sigma() {
    return this.#sigma
  }

  set bias(bias: number) {
    this.#bias = bias
  }

  get bias() {
    return this.#bias
  }

  output(): number {
    if (this.#output !== null) {
      return this.#output
    }

    const sum = this.#synapses.reduce((accumulator, synapse) => {
      return accumulator + (synapse.weight * synapse.neuron.output())
    }, this.#bias)

    return (this.#output = this.#func.compute(sum))
  }

  derivative() {
    return this.#func.derivative(this.output())
  }
}
