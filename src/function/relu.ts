export default class Relu implements FunctionInterface {
  static #instance: Relu

  compute(x: number): number {
    return Math.max(0, x)
  }

  derivative(x: number): number {
    if (x <= 0) {
      return 0
    }

    return 1
  }

  static instance(): Relu {
    this.#instance ??= new this();

    return this.#instance;
  }
}
