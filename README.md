# ts-ml

## Setup

Uses Vite

```bash
yarn install
yarn dev
```

## Create a network

```ts
const network = new Network(
  // Inputs
  2,
  // Output
  { neurons: 1, activationFunction: ActivationFunction.Sigmoid },
  // Hidden layers
  [
    { neurons: 3, activationFunction: ActivationFunction.Relu, bias: 1 },
    { neurons: 3, bias: 0.5 } // Default function is Sigmoid
  ]
)
```

## Train network

```ts
const trainer = new Backpropagation(network, 0.3 /* Learning Rate */)

const xorSamples: TrainingSample[] = [
  { inputs: [0, 0], outputs: [0] },
  { inputs: [0, 1], outputs: [1] },
  { inputs: [1, 0], outputs: [1] },
  { inputs: [1, 1], outputs: [0] },
];
```

### Train one sample

```ts
trainer.train(xorSamples[0])
```

### Train batch (randomized)

```ts
trainer.trainBatch(xorSamples)
```

## Compute

```ts
network.compute(xorSamples[0].inputs)
```

## Importing and Exporting Networks

### Export

Exporting the network will maintain all weights, activation functions, and biases when importing again.

```ts
const networkExport = network.export()
const networkJson = JSON.stringify(networkExport)
```

### Import

```ts
const networkExport = JSON.parse(networkJson)
const network = Network.fromNetworkExport(networkExport)
```
