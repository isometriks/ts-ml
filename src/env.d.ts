import Synapse from "./neuron/synapse.ts";

declare global {
  interface NeuronInterface {
    output(): number
    unCache(): void
    derivative(): number
    addReverseSynapseNeuron(neuron: NeuronInterface, synapse: Synapse)
    set sigma(sigma: number)
    get sigma(): number
    get reverseSynapseNeurons(): [NeuronInterface, Synapse][]
    get identifier(): string
    get label(): string
  }

  interface ConnectableNeuronInterface extends NeuronInterface {
    addSynapse(neuron: NeuronInterface, weight?: number)
    get synapses(): Synapse[]
  }

  interface FunctionInterface {
    compute(x: number): number
    derivative(x: number): number
  }
}
