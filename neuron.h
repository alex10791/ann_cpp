
#include <vector>
#include <iostream>
#include <math.h>

class Neuron {
/*	
	typedef struct inputs {
		double x;
		double w;
		in
	} inputs;
*/	 


private:
	int numberOfInputs;

	std::vector<double> w;
	std::vector<double> old_w;
	std::vector<double> x;
	double expected;

	double lamda;
	//double t;
	double relativeError;

	double activation;
	double output;
	std::vector<double> adjustment;

private:
	double activationFunction();
	double outputFunction();
	void adjustmentFunction();
	void adjustWeights();
	void updatePredictionSet(std::vector<double> x);

public:
	Neuron(int numberOfInputs, double lamda, std::vector<double> w);
	void updateTrainingSet(std::vector<double> x, double y);
	void trainStep();
	double predict(std::vector<double> x);
	void printWeightFunction();
	void updateRelativeError(double partial_error);
	double getNextLayerPartialError(int in);
	void forwardTraining();
	void backwordTraining();
	void clearRelativeError();
	double getOutput();

};

