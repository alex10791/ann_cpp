

#include <iostream>
//#include "perceptron.h"
#include "neuron.h"

using namespace std;


int main() {

/*
	cout << n1.activationFunction() << endl;
	cout << n1.outputFunction() << endl;
	n1.adjustmentFunction();
	n1.adjustWeights();
*/ 


	double xs[8][3];
	double ys[8][2];

	xs[0][0] = 1.0;
	xs[0][1] = 0.0;
	xs[0][2] = 0.0;
	ys[0][0] = -1;
	ys[0][1] = 1;

	xs[1][0] = 1.0;
	xs[1][1] = 0.0;
	xs[1][2] = 1.0;
	ys[1][0] = 1;
	ys[1][1] = -1;
	
	xs[2][0] = 1.0;
	xs[2][1] = 1.0;
	xs[2][2] = 0.0;
	ys[2][0] = 1;
	ys[2][1] = -1;

	xs[3][0] = 1.0;
	xs[3][1] = 1.0;
	xs[3][2] = 1.0;
	ys[3][0] = 1;
	ys[3][1] = -1;

	xs[4][0] = 0.0;
	xs[4][1] = 0.0;
	xs[4][2] = 1.0;
	ys[4][0] = -1;
	ys[4][1] = 1;

	xs[5][0] = 0.0;
	xs[5][1] = 1.0;
	xs[5][2] = 0.0;
	ys[5][0] = -1;
	ys[5][1] = 1;

	xs[6][0] = 0.0;
	xs[6][1] = 1.0;
	xs[6][2] = 1.0;
	ys[6][0] = 1;
	ys[6][1] = -1;

	xs[7][0] = 0.0;
	xs[7][1] = 0.0;
	xs[7][2] = 0.0;
	ys[7][0] = -1;
	ys[7][1] = 1;


/*
	xs[0][0] = 0.0;
	xs[0][1] = 0.0;
	xs[0][2] = 0.0;
	ys[0] = -1;
	xs[1][0] = 1.0;
	xs[1][1] = 0.0;
	xs[1][2] = 0.0;
	ys[1] = -1;
	xs[2][0] = 0.0;
	xs[2][1] = 1.0;
	xs[2][2] = 0.0;
	ys[2] = -1;
	xs[3][0] = 1.0;
	xs[3][1] = 1.0;
	xs[3][2] = 0.0;
	ys[3] = 1;
*/


	vector<double> xs_vect;
	for (int i = 0; i < 3; ++i) {
		xs_vect.push_back(0.0);
	}
	vector<double> ws_vect1;
	for (int i = 0; i < 4; ++i) {
		ws_vect1.push_back(0.0);
	}
	ws_vect1[0] = -0.10;
	ws_vect1[1] = 0.20;
	ws_vect1[2] = 0.30;
	ws_vect1[3] = 0.25;

	vector<double> ws_vect2;
	for (int i = 0; i < 4; ++i) {
		ws_vect2.push_back(0.0);
	}
	ws_vect2[0] = -0.13;
	ws_vect2[1] = 0.24;
	ws_vect2[2] = 0.33;
	ws_vect2[3] = 0.21;

	vector<double> ws_vect3;
	for (int i = 0; i < 4; ++i) {
		ws_vect3.push_back(0.0);
	}
	ws_vect3[0] = -0.15;
	ws_vect3[1] = 0.29;
	ws_vect3[2] = 0.35;
	ws_vect3[3] = 0.26;

	vector<double> ws_vect4;
	for (int i = 0; i < 4; ++i) {
		ws_vect4.push_back(0.0);
	}
	ws_vect4[0] = -0.11;
	ws_vect4[1] = 0.22;
	ws_vect4[2] = 0.37;
	ws_vect4[3] = 0.26;


	Neuron n1 = Neuron(3, 0.2, ws_vect1);
	Neuron n2 = Neuron(3, 0.2, ws_vect2);
	Neuron n3 = Neuron(2, 0.2, ws_vect3);
	Neuron n4 = Neuron(2, 0.2, ws_vect4);

	int kokos;

	for (int j = 0; j < 100000; ++j) {
		for (int i = 0; i < 8; ++i) {


			//cin >> kokos;
			//kokos += kokos + 1;

			xs_vect[0] = xs[i][0];
			xs_vect[1] = xs[i][1];
			xs_vect[2] = xs[i][2];

	// FORWARD TRAINING
			// layers 1 inputs
			n1.updateTrainingSet(xs_vect, ys[i][0]);
			n2.updateTrainingSet(xs_vect, ys[i][1]);

			// layer 1 forward training
			n1.forwardTraining();
			n2.forwardTraining();
			
			// layer 2 inputs
			vector<double> out1;
			out1.push_back(0.0);
			out1.push_back(0.0);
			out1[0] = n1.getOutput();
			out1[1] = n2.getOutput();
			n3.updateTrainingSet(out1, ys[i][0]);

			vector<double> out2;
			out2.push_back(0.0);
			out2.push_back(0.0);
			out2[0] = n1.getOutput();
			out2[1] = n2.getOutput();
			n4.updateTrainingSet(out2, ys[i][1]);

			// layer 2 forward training
			n3.forwardTraining();
			n4.forwardTraining();

			//cout << "n1.getOutput() : " << n1.getOutput() << endl;
			//cout << "n2.getOutput() : " << n2.getOutput() << endl;
			//cout << "n3.getOutput() : " << n3.getOutput() << endl;
			//cout << "n4.getOutput() : " << n4.getOutput() << endl;

	// BACKWORD TRAINING
			// layer 2 update relative error
			n3.clearRelativeError();
			n3.updateRelativeError(ys[i][0] - n3.getOutput());
			n4.clearRelativeError();
			n4.updateRelativeError(ys[i][1] - n4.getOutput());

			// layer 2 adjust weights
			n3.backwordTraining();
			n4.backwordTraining();


			// layer 1 update relative error
			n1.clearRelativeError();
			//n1.updateRelativeError(ys[i] - n1.getOutput());
			//cout << "n3.getNextLayerPartialError(0) : " << n3.getNextLayerPartialError(0) << endl;
			//cout << "n4.getNextLayerPartialError(0) : " << n4.getNextLayerPartialError(0) << endl;
			//cout << "n3.getNextLayerPartialError(1) : " << n3.getNextLayerPartialError(1) << endl;
			//cout << "n4.getNextLayerPartialError(1) : " << n4.getNextLayerPartialError(1) << endl;

			n1.updateRelativeError(n3.getNextLayerPartialError(0));
			n1.updateRelativeError(n4.getNextLayerPartialError(0));
			n2.clearRelativeError();
			n2.updateRelativeError(n3.getNextLayerPartialError(1));	
			n2.updateRelativeError(n4.getNextLayerPartialError(1));		

			// layer 1 adjust weights
			n1.backwordTraining();
			n2.backwordTraining();



	// Train neuron4 seperately
//			n4.updateTrainingSet(xs_vect, ys[i]);
			//n4.trainStep();
//			n4.forwardTraining();
			//cout << p1.getErrorDifference() << endl;
//			n4.clearRelativeError();
//			n4.updateRelativeError(ys[i] - n4.getOutput());
			//n1.getNextLayerPartialError(0);
//			n4.backwordTraining();
			//cin >> kokos;
		}
	}

	cout << "Kokos2" << endl;
	
	n1.printWeightFunction();
	n2.printWeightFunction();
	n3.printWeightFunction();
	n4.printWeightFunction();

	
	for (int i = 0; i < 8; ++i) {
		xs_vect[0] = xs[i][0];
		xs_vect[1] = xs[i][1];
		xs_vect[2] = xs[i][2];
		vector<double> out3;
		out3.push_back(0.0);
		out3.push_back(0.0);
		out3[0] = n1.predict(xs_vect);
		out3[1] = n2.predict(xs_vect);
		cout << n3.predict(out3) << "\t";
		cout << n4.predict(out3) << endl;
	}

/*	
	cout << n1.predict(1.0, 0.0, 0.0) << endl;
	cout << n1.predict(1.0, 0.0, 1.0) << endl;
	cout << n1.predict(1.0, 1.0, 1.0) << endl;
	cout << n1.predict(0.0, 0.0, 0.0) << endl;
*/


	return 0;
}



