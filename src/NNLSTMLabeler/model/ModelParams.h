#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

  public:
    Alphabet wordAlpha; // should be initialized outside
    LookupTable words; // should be initialized outside
    Alphabet featAlpha;
    LSTM1Params lstm_left_param;
    LSTM1Params lstm_right_param;
    UniParams olayer_linear; // output
  public:
    Alphabet labelAlpha; // should be initialized outside
    SoftMaxLoss loss;


  public:
    bool initial(HyperParams& opts) {

        // some model parameters should be initialized outside
        if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
            return false;
        }
        opts.wordDim = words.nDim;
        opts.wordWindow = opts.wordContext * 2 + 1;
        opts.windowOutput = opts.wordDim * opts.wordWindow;
        opts.labelSize = labelAlpha.size();
        lstm_left_param.initial(opts.hiddenSize, opts.windowOutput);
        lstm_right_param.initial(opts.hiddenSize, opts.windowOutput);
        opts.inputSize = opts.hiddenSize * 3 * 2;
        olayer_linear.initial(opts.labelSize, opts.inputSize, false);
        return true;
    }

    bool TestInitial(HyperParams& opts) {

        // some model parameters should be initialized outside
        if (words.nVSize <= 0 || labelAlpha.size() <= 0) {
            return false;
        }
        opts.wordDim = words.nDim;
        opts.wordWindow = opts.wordContext * 2 + 1;
        opts.windowOutput = opts.wordDim * opts.wordWindow;
        opts.labelSize = labelAlpha.size();
        opts.inputSize = opts.hiddenSize * 3;
        return true;
    }

    void exportModelParams(ModelUpdate& ada) {
        words.exportAdaParams(ada);
        lstm_left_param.exportAdaParams(ada);
        lstm_right_param.exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }


    void exportCheckGradParams(CheckGrad& checkgrad) {
		checkgrad.add(&words.E, "words E");
		//left lstm
		checkgrad.add(&lstm_left_param.cell.W1, "lstm_left_param.cell.W1");
		checkgrad.add(&lstm_left_param.cell.W2, "lstm_left_param.cell.W2");
		checkgrad.add(&lstm_left_param.cell.b, "lstm_left_param.cell.b");
		checkgrad.add(&lstm_left_param.forget.W1, "lstm_left_param.forget.W1");
		checkgrad.add(&lstm_left_param.forget.W2, "lstm_left_param.forget.W2");
		checkgrad.add(&lstm_left_param.forget.b, "lstm_left_param.forget.b");
		checkgrad.add(&lstm_left_param.input.W1, "lstm_left_param.input.W1");
		checkgrad.add(&lstm_left_param.input.W2, "lstm_left_param.input.W2");
		checkgrad.add(&lstm_left_param.input.b, "lstm_left_param.input.b");
		checkgrad.add(&lstm_left_param.output.W1, "lstm_left_param.output.W1");
		checkgrad.add(&lstm_left_param.output.W2, "lstm_left_param.output.W2");
		checkgrad.add(&lstm_left_param.output.b, "lstm_left_param.output.b");
		checkgrad.add(&lstm_right_param.cell.W1, "lstm_right_param.cell.W1");
		checkgrad.add(&lstm_right_param.cell.W2, "lstm_right_param.cell.W2");
		checkgrad.add(&lstm_right_param.cell.b, "lstm_right_param.cell.b");
		//right lstm
		checkgrad.add(&lstm_right_param.forget.W1, "lstm_right_param.forget.W1");
		checkgrad.add(&lstm_right_param.forget.W2, "lstm_right_param.forget.W2");
		checkgrad.add(&lstm_right_param.forget.b, "lstm_right_param.forget.b");
		checkgrad.add(&lstm_right_param.input.W1, "lstm_right_param.input.W1");
		checkgrad.add(&lstm_right_param.input.W2, "lstm_right_param.input.W2");
		checkgrad.add(&lstm_right_param.input.b, "lstm_right_param.input.b");
		checkgrad.add(&lstm_right_param.output.W1, "lstm_right_param.output.W1");
		checkgrad.add(&lstm_right_param.output.W2, "lstm_right_param.output.W2");
		checkgrad.add(&lstm_right_param.output.b, "lstm_right_param.output.b");

		checkgrad.add(&olayer_linear.W, "output layer W");
    }

    // will add it later
    void saveModel(std::ofstream &os) const {
    }

    void loadModel(std::ifstream &is) {
    }

};

#endif /* SRC_ModelParams_H_ */