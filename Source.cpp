#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <ctime>
#include <iomanip>
#include <fstream>

using std::vector;
using std::ifstream;
using std::cout;

enum { SYGMOID, HYPTAN };

void Sum(const vector<int>& enters, const vector<vector<double>>&	synapses_hidden,
	const vector<vector<double>>& synapses_output, vector<double>& hidden_layer, vector<double>& outputs, const int& funcType);
double GetRandDouble();
double DerivetedFunc(const double& x, int funcType);
double TransfFunc(const double& x, int funcType);

int main() {
	srand(time(NULL));
	ifstream fin("Data.txt");
	if (!fin.is_open() || fin.eof()) {
		cout << "File is not open or empty";
		return 0;
	}

	// get number of inputs, iterations, hidden neurons, outputs and activation function
	std::string strTmp;
	std::stringstream strStream;
	getline(fin, strTmp, ';');
	strStream << strTmp;
	int numHidLayers = 1, numHidNeurons, numInputs, numOutputs, numIterations, funcType;
	strStream >> numInputs >> numIterations >> numHidNeurons >> numOutputs >> funcType;
	strStream.clear();

	vector<int> inputs(numInputs + 1, 1);
	vector<double> hidLayer(numHidNeurons + 1, 0);
	hidLayer[numHidNeurons] = 1; // this neuron on hidden layer is bias

	vector<vector<int>> learn(0, vector<int>(numInputs));
	vector<vector<int>> learnAnsw(0, vector<int>(numOutputs));

	vector<int> tmp(numInputs, 0);
	getline(fin, strTmp, ';');
	strStream << strTmp;
	while (!strStream.eof()) {
		for (int j = 0; j < numInputs; ++j) {
			strStream >> tmp[j];
		}
		learn.push_back(tmp);
	}
	strStream.clear();

	tmp.resize(numOutputs);
	getline(fin, strTmp, ';');
	strStream << strTmp;
	while (!strStream.eof()) {
		for (int j = 0; j < numOutputs; ++j) {
			strStream >> tmp[j];
		}
		learnAnsw.push_back(tmp);
	}
	strStream.clear();

	// initialize weights with random values
	vector<vector<double>> synapsesHid(numHidNeurons, vector<double>(numInputs + 1, 0));
	for (int i = 0; i < numHidNeurons; ++i) {
		for (int j = 0; j < numInputs + 1; ++j) {
			synapsesHid[i][j] = GetRandDouble();
		}
	}
	vector<vector<double>> synapsesOut(numOutputs, vector<double>(numHidNeurons + 1, GetRandDouble()));
	for (int i = 0; i < numOutputs; ++i) {
		for (int j = 0; j < numHidNeurons + 1; ++j) {
			synapsesOut[i][j] = GetRandDouble();
		}
	}

	vector<double> outputs(numOutputs, 0), errors(numOutputs, 0), outGradietns(numOutputs, 0);
	vector<double> hiddenGradients(numHidNeurons, 0);
	vector<double> gErrors(numIterations, 0); // слой ошибок

	int countIter = 0;
	double gError = 0, learnSpeed = 0.2, deltSynp = 0;

	do {
		++countIter;
		cout << "\nPass: "<< countIter << " Error: ";
		cout << std::fixed << std::setprecision(20) << gError;
		gError = 0;

		for (int p = 0; p < numIterations; p++) {

			for (int i = 0; i < inputs.size() - 1; i++) {
				inputs[i] = learn[p][i]; // provide data to network inputs
			}

			Sum(inputs, synapsesHid, synapsesOut, hidLayer, outputs, funcType); // feed foward

			for (int i = 0; i < outputs.size(); ++i) {
				errors[i] = learnAnsw[p][i] - outputs[i]; // get error
				outGradietns[i] = errors[i] * DerivetedFunc(outputs[i], funcType); // calculate output gradients
			}
			gErrors[p] = 0;
			for (auto i : errors) {
				gErrors[p] += (i * i) / 2; // calculate error for current iteration
			}
			for (int i = 0; i < hidLayer.size() - 1; i++) { // pass the error to the layer of hidden neuron
				for (int j = 0; j < outputs.size(); j++) { // calculate hidden gradients
					hiddenGradients[i] += synapsesOut[j][i] * outGradietns[j] * DerivetedFunc(hidLayer[i], funcType);
				}
			}
			
			// change weights of output neurons
			for (int i = 0; i < hidLayer.size() - 1; i++) {
				for (int j = 0; j < inputs.size(); j++) {
					deltSynp = learnSpeed *  hiddenGradients[i] * inputs[j];
					synapsesHid[i][j] += deltSynp;
				}
			}
			for (int i = 0; i < hidLayer.size() - 1; i++) {
				for (int j = 0; j < outputs.size(); j++) {
					deltSynp += learnSpeed * outGradietns[j] * hidLayer[i];
					synapsesOut[j][i] += deltSynp;
				}
			}

			for (int i = 0; i < numHidNeurons; i++) {
				hiddenGradients[i] = 0;
			}
			cout << std::endl;
			for (int i = 0; i < numInputs; i++) {
				cout << inputs[i] << "  ";
			}
			for (int i = 0; i < numOutputs; i++) {
				cout << outputs[i] << "  ";
			}
		}

		for (auto i : gErrors) {
			gError += (i * i) / 2; // calculate global error for whole age 
		}
		cout << "\n\n";
	} while (gError > 0.01);

	system("pause");
	return 0;
}

void Sum(	const vector<int>&		inputs,
	const vector<vector<double>>&	synapsesHid,
	const vector<vector<double>>&	synapsesOut,
	vector<double>&					hidLayer,
	vector<double>&					outputs,
	const int&						funcType) {
	// start distribution of a signal
	double output = 0;
	for (int i = 0; i < hidLayer.size() - 1; i++) {
		hidLayer[i] = 0;
		for (int j = 0; j < inputs.size(); j++) {
			hidLayer[i] += synapsesHid[i][j] * inputs[j]; // summarize
		}
		hidLayer[i] = TransfFunc(hidLayer[i], funcType); // pass hidden neuron through the activation function
	}

	for (int i = 0; i < outputs.size(); i++) {
		for (int j = 0; j < hidLayer.size(); j++) {
			outputs[i] += synapsesOut[i][j] * hidLayer[j]; // calculate output
		}
		outputs[i] = TransfFunc(outputs[i], funcType); // pass output neuron through the activation function
	}
}

double GetRandDouble() {
	return (rand() % 101) * 0.02 - 1;
}

double DerivetedFunc(const double& x, int funcType) {
	switch (funcType) {
	case SYGMOID: return x * (1 - x);
	case HYPTAN: return (1 - x * x);
	default: std::cerr << "Incorrect value of activation function!";
		exit(0);
	}
}

double TransfFunc(const double& x, int funcType) {
	switch (funcType) {
	case SYGMOID: return 1 / (1 + exp(-x));
	case HYPTAN: return tanh(x);
	default: std::cerr << "Incorrect value of activation function!";
		exit(0);
	}
}