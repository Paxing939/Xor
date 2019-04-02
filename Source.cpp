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
	ifstream fin("Data.txt");
	if (!fin.is_open() || fin.eof()) {
		cout << "File is not open or empty";
		return 0;
	}

	int funcType;
	std::string strTmp;
	std::stringstream strStream;
	getline(fin, strTmp, ';');
	strStream << strTmp;

	srand(time(NULL));
	int numHidLayers = 1, numHidNeurons = 4, numEnters = 2, numOutputs = 2, numAges = 4;
	int numb_synapses_hidden = numHidNeurons * numEnters;
	strStream >> numEnters >> numHidLayers >> numAges >> numHidNeurons >> numOutputs >> funcType;
	strStream.clear();

	vector<int> enters(numEnters + 1, 1);
	vector<double> hidLayer(numHidNeurons + 1, 0);
	hidLayer[numHidNeurons] = 1;

	vector<vector<int>> learnAnsw(0, vector<int>(numOutputs));
	vector<vector<int>> learn(0, vector<int>(numEnters));

	vector<int> tmp(numEnters, 0);
	getline(fin, strTmp, ';');
	strStream << strTmp;
	while (!strStream.eof()) {
		for (int j = 0; j < numEnters; ++j) {
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

	vector<vector<double>> synapsesHid(numHidNeurons, vector<double>(numEnters + 1, 0));
	for (int i = 0; i < numHidNeurons; ++i) {
		for (int j = 0; j < numEnters + 1; ++j) {
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
	vector<double> gErrors(numAges, 0); // слой ошибок

	int countIter = 0;
	double gError = 0, lernSpeed = 0.2, momentum = 0.5, deltSynp = 0;

	do {
		++countIter;
		std::cout << "\nPass: "<< countIter << " Error: ";
		std::cout << std::fixed << std::setprecision(20) << gError;
		gError = 0; // обнуляем

		for (int p = 0; p < numAges; p++) {

			for (int i = 0; i < enters.size() - 1; i++) {
				enters[i] = learn[p][i]; // подаём данные на входы сети
			}

			Sum(enters, synapsesHid, synapsesOut, hidLayer, outputs, funcType);

			for (int i = 0; i < outputs.size(); ++i) {
				errors[i] = learnAnsw[p][i] - outputs[i]; // получаем ошибку
				outGradietns[i] = errors[i] * DerivetedFunc(outputs[i], funcType);
			}
			gErrors[p] = 0;
			for (auto i : errors) {
				gErrors[p] += (i * i) / 2;
			}
			for (int i = 0; i < hidLayer.size() - 1; i++) { // передаём ошибку на второй слой ошибок скрытых нейронов
				for (int j = 0; j < outputs.size(); j++) {
					hiddenGradients[i] += synapsesOut[j][i] * outGradietns[j] * DerivetedFunc(hidLayer[i], funcType);
				}
			} // по связям к выходу
			

			for (int i = 0; i < hidLayer.size() - 1; i++) {
				for (int j = 0; j < enters.size(); j++) {
					//new_delt_synp = speed_lerning * hidden_layer1[i] * hidden_gradients[i] + momentum * old_delt_synp;
					deltSynp = lernSpeed *  hiddenGradients[i] * enters[j];
					synapsesHid[i][j] += deltSynp;
				}
			}
			for (int i = 0; i < hidLayer.size() - 1; i++) {
				//new_delt_synp = speed_lerning * output * gradientOut + momentum * old_delt_synp;
				for (int j = 0; j < outputs.size(); j++) {
					deltSynp += lernSpeed * outGradietns[j] * hidLayer[i];
					synapsesOut[j][i] += deltSynp;
				}
			}// меняем веса выходных нейронов

			for (int i = 0; i < numHidNeurons; i++) {
				hiddenGradients[i] = 0;
			}
			std::cout << std::endl;
			for (int i = 0; i < numEnters; i++) {
				std::cout << enters[i] << "  ";
			}
			for (int i = 0; i < numOutputs; i++) {
				std::cout << outputs[i] << "  ";
			}
		}

		for (auto i : gErrors) {
			gError += (i * i) / 2;
		}
		std::cout << "\n\n";
	} while (gError > 0.01);

	system("pause");
	return 0;
}

void Sum(	const vector<int>&		enters,
	const vector<vector<double>>&	synapsesHid,
	const vector<vector<double>>&	synapsesOut,
	vector<double>&					hidLayer,
	vector<double>&					outputs,
	const int&						funcType) {
	// запускаем распространение сигнала
	double output = 0;
	for (int i = 0; i < hidLayer.size() - 1; i++) {
		hidLayer[i] = 0;
		for (int j = 0; j < enters.size(); j++) {
			hidLayer[i] += synapsesHid[i][j] * enters[j]; // суммируем
		}
		hidLayer[i] = TransfFunc(hidLayer[i], funcType); // пропускаем скрытый нейрон через функцию активации
	}

	for (int i = 0; i < outputs.size(); i++) {
		for (int j = 0; j < hidLayer.size(); j++) {
			outputs[i] += synapsesOut[i][j] * hidLayer[j]; // считаем выход
		}
		outputs[i] = TransfFunc(outputs[i], funcType); // пропускаем output через th
	}
}

double GetRandDouble() {
	return (rand() % 101) * 0.02 - 1;
}

double DerivetedFunc(const double& x, int funcType) {
	switch (funcType) {
	case SYGMOID: return x * (1 - x);
	case HYPTAN: return (1 - x * x);
	}
}

double TransfFunc(const double& x, int funcType) {
	switch (funcType) {
	case SYGMOID: return 1 / (1 + exp(-x));
	case HYPTAN: return tanh(x);
	}
}