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
  { neurons: 1, activationFunction: Sigmoid.instance() },
  // Hidden layers
  [
    { neurons: 3, activationFunction: Sigmoid.instance(), bias: 1 },
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
