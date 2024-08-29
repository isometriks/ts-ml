export default class Sigmoid implements FunctionInterface {
  static #instance: Sigmoid

  compute(x: number): number {
    return 1 / (1 + Math.exp(-x))
  }

  derivative(x: number): number {
    const sigmoid = this.compute(x)
    return sigmoid * (1 - sigmoid)
  }

  static instance(): Sigmoid {
    if (!this.#instance) {
      this.#instance = new this();
    }

    return this.#instance;
  }
}
