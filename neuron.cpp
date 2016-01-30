
#include "neuron.h"

Neuron::Neuron(int numberOfInputs, double lamda, std::vector<double> w) {

	this->numberOfInputs = numberOfInputs + 1;

	this->x = std::vector<double>();
	this->w = std::vector<double>();
	this->adjustment = std::vector<double>();

	for (int i = 0; i < this->numberOfInputs; ++i) {
		this->x.push_back(0.0);
		this->w.push_back(0.0);
		this->old_w.push_back(0.0);
		this->adjustment.push_back(0.0);
	}

	this->x[0] = 1.0;

	this->expected = -1;

	int size_arg_w = w.size();
	for (int i = 0; i < this->numberOfInputs; ++i) {
		if (i < size_arg_w) {
			this->w[i] = w[i];
		} else {
			this->w[i] = 0.0;
		}
		
	}

	this->lamda = lamda;
	this->relativeError = 0.0;
	//this->t = t;

}

double Neuron::activationFunction() {
	double nominator = 0.0;
	//double denominator = 0.0;
	for (int i = 0; i < this->numberOfInputs; ++i) {
		nominator += x[i]*w[i];
		//denominator += w[i];
	}
	this->activation = nominator;	///denominator;

	return activation;
}

double Neuron::outputFunction() {
	this->output = 1/(1+exp(this->activation));
	return this->output;
}

void Neuron::adjustmentFunction() {
	double adjustment_const = -2.0*this->lamda * this->relativeError * this->output * (1-this->output);
	//double adjustment_const = -2.0*this->lamda * (this->expected-this->output) * this->output * (1-this->output);
	//double adjustment_const = this->lamda * (this->expected-this->output);
	for (int i = 0; i < this->numberOfInputs; ++i) {
		this->adjustment[i] = adjustment_const * this->x[i];
	} 
}

void Neuron::adjustWeights() {
	for (int i = 0; i < this->numberOfInputs; ++i) {
		this->old_w[i] = this->w[i];
		this->w[i] += this->adjustment[i];
	}
}

void Neuron::trainStep() {
	activationFunction();
	outputFunction();
	adjustmentFunction();
	adjustWeights();
}

void Neuron::forwardTraining() {
	activationFunction();
	outputFunction();
}

void Neuron::backwordTraining() {
	adjustmentFunction();
	adjustWeights();
}

void Neuron::updateTrainingSet(std::vector<double> x, double y) {
	int size_arg_x = x.size();
	for (int i = 1; i < this->numberOfInputs; ++i) {
		if (i < size_arg_x+1) {
			this->x[i] = x[i-1];
		} else {
			this->x[i] = 0.0;
		}
		
	}
	this->expected = y;
}

void Neuron::updatePredictionSet(std::vector<double> x) {
	int size_arg_x = x.size();
	for (int i = 1; i < this->numberOfInputs; ++i) {
		if (i < size_arg_x+1) {
			this->x[i] = x[i-1];
		} else {
			this->x[i] = 0.0;
		}
		
	}
}

double Neuron::predict(std::vector<double> x) {
	updatePredictionSet(x);
	activationFunction();
	return outputFunction();
}

void Neuron::printWeightFunction() {
	for (int i = 0; i < this->numberOfInputs; ++i) {
		std::cout << "w[" << i << "] : " << w[i] << std::endl;
	}
	std::cout << "y = ";
	for (int i = 1; i < this->numberOfInputs; ++i) {
		std::cout << w[i] << "x" << i << "+";
	}
	std::cout << w[0] << std::endl;
}

void Neuron::updateRelativeError(double partialError) {
	this->relativeError += partialError;
}

double Neuron::getNextLayerPartialError(int in) {
	if (in < numberOfInputs-1) {
		return -1.0 * this->relativeError * this->output * (1 - this->output) * this->old_w[in+1];		//-2.0*this->lamda * this->relativeError * this->output * (1 - this->output) * this->old_w[in+1];
	}
	std::cout << "NEURON DOES NOT EXIST" << std::endl;
	return 0.0;
}

void Neuron::clearRelativeError() {
	this->relativeError = 0.0;
}

double Neuron::getOutput() {
	return this->output;
}
