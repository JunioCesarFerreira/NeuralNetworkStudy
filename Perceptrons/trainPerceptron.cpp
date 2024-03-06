#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class Neuron {
private:
    vector<double> weights;
    double output;
    double error;

    // Função de ativação sigmoid
    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }

    // Derivada da função sigmoid para retropropagação
    double sigmoid_derivative(double x) {
        return x * (1 - x);
    }

public:
    Neuron(int num_inputs) {
        // Inicializa pesos aleatórios entre -1 e 1
        for (int i = 0; i <= num_inputs; ++i) { // Inclui bias
            weights.push_back((double)rand() / RAND_MAX * 2 - 1);
        }
    }

    // Calcula a saída do neurônio
    double calculate_output(const vector<double>& inputs) {
        double sum = 0;
        for (int i = 0; i < inputs.size(); ++i) {
            sum += inputs[i] * weights[i];
        }
        output = sigmoid(sum);
        return output;
    }

    // Retorna a saída do neurônio
    double get_output() const {
        return output;
    }

    // Calcula o erro do neurônio na camada de saída
    void calculate_output_error(double target) {
        error = target - output;
    }

    // Calcula o erro do neurônio nas camadas ocultas
    void calculate_hidden_error(const vector<double>& next_layer_errors, const vector<double>& next_layer_weights) {
        double sum = 0;
        for (int i = 0; i < next_layer_errors.size(); ++i) {
            sum += next_layer_errors[i] * next_layer_weights[i];
        }
        error = sum * sigmoid_derivative(output);
    }

    // Atualiza os pesos do neurônio
    void update_weights(const vector<double>& inputs, double learning_rate) {
        for (int i = 0; i < weights.size(); ++i) {
            weights[i] += learning_rate * error * inputs[i];
        }
    }
        
    // Função para obter os pesos do neurônio
    vector<double> get_weights() const {
        return weights;
    }
};

class Layer {
private:
    vector<Neuron> neurons;

public:
    Layer(int num_neurons, int num_inputs_per_neuron) {
        for (int i = 0; i < num_neurons; ++i) {
            neurons.push_back(Neuron(num_inputs_per_neuron));
        }
    }

    // Calcula as saídas da camada
    vector<double> calculate_outputs(const vector<double>& inputs) {
        vector<double> outputs;
        for (int i = 0; i < neurons.size(); ++i) {
            outputs.push_back(neurons[i].calculate_output(inputs));
        }
        return outputs;
    }

    // Retorna as saídas da camada
    vector<double> get_outputs() const {
        vector<double> outputs;
        for (int i = 0; i < neurons.size(); ++i) {
            outputs.push_back(neurons[i].get_output());
        }
        return outputs;
    }

    // Atualiza os pesos da camada
    void update_weights(const vector<double>& inputs, double learning_rate) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].update_weights(inputs, learning_rate);
        }
    }

    vector<vector<double>> get_weights() const {
        vector<vector<double>> weights;
        for (const auto& neuron : neurons) {
            // Supondo que exista uma função get_weights em Neuron que retorna os pesos do neurônio
            weights.push_back(neuron.get_weights());
        }
        return weights;
    }

    // Calcula os erros dos neurônios na camada de saída
    void calculate_output_errors(const vector<double>& targets) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].calculate_output_error(targets[i]);
        }
    }

    // Calcula os erros dos neurônios nas camadas ocultas
    void calculate_hidden_errors(const vector<double>& next_layer_errors, const vector<vector<double>>& next_layer_weights) {
        for (int i = 0; i < neurons.size(); ++i) {
            neurons[i].calculate_hidden_error(next_layer_errors, next_layer_weights[i]);
        }
    }
};

class NeuralNetwork {
private:
    vector<Layer> layers;
    double learning_rate;

public:
    NeuralNetwork(double rate, vector<int> topology) : learning_rate(rate) {
        srand(static_cast<unsigned>(time(0))); // Inicialização do gerador de números aleatórios
        int num_inputs = topology[0];
        for (size_t i = 1; i < topology.size(); ++i) {
            layers.emplace_back(topology[i], num_inputs);
            num_inputs = topology[i];
        }
    }

    // Treina a rede neural
    void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (int i = 0; i < inputs.size(); ++i) {
                vector<double> input = inputs[i];
                feed_forward(input);
                backpropagate(targets[i]);
            }
        }
    }

    // Alimenta a entrada pela rede
    void feed_forward(const vector<double>& input) {
        vector<double> tmp = input;
        for (auto& layer : layers) {
            tmp = layer.calculate_outputs(tmp);
        }
    }

    // Retropropagação do erro
    void backpropagate(const vector<double>& targets) {
        // Calcula erros da camada de saída
        layers.back().calculate_output_errors(targets);

        // Calcula erros das camadas ocultas
        for (int i = layers.size() - 2; i >= 0; --i) {
            layers[i].calculate_hidden_errors(
                layers[i].get_outputs(), layers[i].get_weights()
                );
        }

        // Atualiza os pesos
        vector<double> input = layers[0].get_outputs();
        for (int i = 0; i < layers.size(); ++i) {
            layers[i].update_weights(input, learning_rate);
            input = layers[i].get_outputs();
        }
    }

    // Função para obter a saída da rede neural
    vector<double> get_output() const {
        if (!layers.empty()) {
            return layers.back().get_outputs();
        }
        // Retorna um vetor vazio se não houver camadas, indicando que não há saída
        return {};
    }
};

int main() {
    cout << "Neuron Test" << endl;
    Neuron neuron(6);
    vector<double> inputTest = {0, 1, 0, 2, 1, 0};
    cout << "result: " << neuron.calculate_output(inputTest) << endl;

    cout << "Layer Test" << endl;
    Layer layer(3, 6);
    layer.calculate_outputs(inputTest);
    for (auto& t : layer.get_outputs()){
        cout << t << endl;
    }

    cout << "Teste da Rede Neural:" << endl;
    // Define a topologia da rede neural: 2 entradas, 2 neurônios na camada oculta, 1 neurônio na camada de saída
    vector<int> topology = {2, 2, 1};
    double learning_rate = 0.1;
    NeuralNetwork neural_network(learning_rate, topology);

    // Define os dados de treinamento
    vector<vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}};

    // Treina a rede neural por 10000 épocas
    neural_network.train(inputs, targets, 10000);

    // Testa a rede neural
    for (int i = 0; i < inputs.size(); ++i) {
        vector<double> input = inputs[i];
        neural_network.feed_forward(input);
        vector<double> output = neural_network.get_output();
        cout << "in: (" << input[0] << ", " << input[1] << ") out: " << output[0] << endl;
    }

    return 0;
}
