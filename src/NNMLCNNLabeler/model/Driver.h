/*
* Driver.h
*
*  Created on: Mar 18, 2015
*      Author: mszhang
*/

#ifndef SRC_Driver_H_
#define SRC_Driver_H_

#include <iostream>
#include "ComputionGraph.h"

//A native neural network classfier using only word embeddings

class Driver {
  public:
    Driver() {

    }

    ~Driver() {

    }

  public:
    Graph _cg;  // build neural graphs
    vector<GraphBuilder> _builders;
    ModelParams _modelparams;  // model parameters
    HyperParams _hyperparams;

    Metric _eval;
    CheckGrad _checkgrad;
    ModelUpdate _ada;  // model update



  public:
    //embeddings are initialized before this separately.
    inline void initial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
            return;
        }
        if (!_modelparams.initial(_hyperparams)) {
            std::cout << "model parameter initialization Error, Please check!" << std::endl;
            return;
        }
        _modelparams.exportModelParams(_ada);
        _modelparams.exportCheckGradParams(_checkgrad);

        _hyperparams.print();

        _builders.resize(_hyperparams.batch);

        for (int idx = 0; idx < _hyperparams.batch; idx++) {
            _builders[idx].createNodes(GraphBuilder::max_sentence_length, _hyperparams.cnnLayerSize);
            _builders[idx].initial(_modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
    }


    inline void TestInitial() {
        if (!_hyperparams.bValid()) {
            std::cout << "hyper parameter initialization Error, Please check!" << std::endl;
            return;
        }
        if (!_modelparams.TestInitial(_hyperparams)) {
            std::cout << "model parameter initialization Error, Please check!" << std::endl;
            return;
        }
        _modelparams.exportModelParams(_ada);
        _modelparams.exportCheckGradParams(_checkgrad);

        _hyperparams.print();

        _builders.resize(_hyperparams.batch);

        for (int idx = 0; idx < _hyperparams.batch; idx++) {
            _builders[idx].createNodes(GraphBuilder::max_sentence_length, _hyperparams.cnnLayerSize);
            _builders[idx].initial(_modelparams, _hyperparams);
        }

        setUpdateParameters(_hyperparams.nnRegular, _hyperparams.adaAlpha, _hyperparams.adaEps);
    }



    inline dtype train(const vector<Example>& examples) {
        _eval.reset();
        int example_num = examples.size();
        if (example_num > _builders.size()) {
            std::cout << "input example number larger than predefined batch number" << std::endl;
            return 1000;
        }

        dtype cost = 0.0;
		_cg.clearValue(true);
        for (int count = 0; count < example_num; count++) {
            const Example& example = examples[count];
            //forward
            _builders[count].forward(&_cg, example.m_feature);

        }
        _cg.compute();

        for (int count = 0; count < example_num; count++) {
            const Example& example = examples[count];
            cost += _modelparams.loss.loss(&_builders[count]._neural_output, example.m_label, _eval, example_num);
        }
        _cg.backward();

        if (_eval.getAccuracy() < 0) {
            std::cout << "strange" << std::endl;
        }

        return cost;
    }

    inline void predict(const vector<Feature>& features, vector<int>& results) {
		results.clear();
		int num = features.size();
		if (num > _builders.size()) {
			std::cout << "input example number is larger than predefined batch number" << std::endl;
			return;
		}
		
		_cg.clearValue();
		for (int idx = 0; idx < num; idx++) {
			_builders[idx].forward(&_cg, features[idx]);
		}
		_cg.compute();

		results.resize(num);
		for (int idx = 0; idx < num; idx++) {
			_modelparams.loss.predict(&_builders[idx]._neural_output, results[idx]);
		}      
    }

    inline dtype cost(const Example& example) {
        _cg.clearValue();
        _builders[0].forward(&_cg, example.m_feature);
        _cg.compute();

        dtype cost = _modelparams.loss.cost(&_builders[0]._neural_output, example.m_label, 1);

        return cost;
    }


    void updateModel() {
        //_ada.update();
        _ada.update(10.0);
    }

    void checkgrad(const vector<Example>& examples, int iter) {
        ostringstream out;
        out << "Iteration: " << iter;
        _checkgrad.check(this, examples, out.str());
    }

    void writeModel();

    void loadModel();



  private:
    inline void resetEval() {
        _eval.reset();
    }


    inline void setUpdateParameters(dtype nnRegular, dtype adaAlpha, dtype adaEps) {
        _ada._alpha = adaAlpha;
        _ada._eps = adaEps;
        _ada._reg = nnRegular;
    }

};

#endif /* SRC_Driver_H_ */
