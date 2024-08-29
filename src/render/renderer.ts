import Network from "../network/network.ts";
import Graph from "graphology";
import Sigma from "sigma";
import Synapse from "../neuron/synapse.ts";

export default class Renderer {
  #container: HTMLElement

  constructor(container: HTMLElement) {
    this.#container = container
  }

  draw(network: Network) {
    const graph = new Graph();
    const neurons: [NeuronInterface, number, number][] = [];
    const edges: [NeuronInterface, NeuronInterface, Synapse][] = [];
    const layers = [network.outputLayer, ...network.hiddenLayers.toReversed(), network.inputLayer]

    layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        neurons.push([neuron, layerIndex, neuronIndex])

        neuron.synapses.forEach((synapse: Synapse) => {
          edges.push([neuron, synapse.neuron, synapse])
        })
      })
    })

    neurons.forEach(([neuron, layerIndex, neuronIndex]) => {
      graph.addNode(neuron.identifier, {
        size: 20,
        label: neuron.label,
        x: layerIndex * -200,
        y: -neuronIndex * 80,
      })
    })

    edges.forEach(([to, from, synapse]: [NeuronInterface, NeuronInterface, Synapse]) => {
      graph.addEdge(from.identifier, to.identifier, {
        type: "arrow",
        label: synapse.weight.toFixed(4),
        size: 3,
      })
    })

    this.#container.innerHTML = "";
    new Sigma(graph, this.#container, {
      renderEdgeLabels: true,
    })
  }
}
