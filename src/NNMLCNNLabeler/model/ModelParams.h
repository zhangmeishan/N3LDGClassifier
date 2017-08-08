#ifndef SRC_ModelParams_H_
#define SRC_ModelParams_H_
#include "HyperParams.h"

// Each model consists of two parts, building neural graph and defining output losses.
class ModelParams {

  public:
    Alphabet wordAlpha; // should be initialized outside
    LookupTable words; // should be initialized outside
    Alphabet featAlpha;
    vector<UniParams> hidden_linear_layers;
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
        opts.windowOutput_layer1 = opts.wordDim * opts.wordWindow;
        opts.windowOutput_layer2 = opts.hiddenSize * opts.wordWindow;

        opts.labelSize = labelAlpha.size();
        hidden_linear_layers.resize(opts.cnnLayerSize);

        hidden_linear_layers[0].initial(opts.hiddenSize, opts.windowOutput_layer1, true);
        int cnnLayerSize = hidden_linear_layers.size();
        for (int idx = 1; idx < cnnLayerSize; idx++) {
            hidden_linear_layers[idx].initial(opts.hiddenSize, opts.windowOutput_layer2, true);
        }

        opts.inputSize = opts.hiddenSize * 3;
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
        opts.windowOutput_layer1 = opts.wordDim * opts.wordWindow;
        opts.labelSize = labelAlpha.size();
        opts.inputSize = opts.hiddenSize * 3;
        return true;
    }

    void exportModelParams(ModelUpdate& ada) {
        words.exportAdaParams(ada);
        int cnnLayerSize = hidden_linear_layers.size();
        for(int idx = 0; idx < cnnLayerSize; idx++)
            hidden_linear_layers[idx].exportAdaParams(ada);
        olayer_linear.exportAdaParams(ada);
    }


    void exportCheckGradParams(CheckGrad& checkgrad) {
        checkgrad.add(&olayer_linear.W, "output layer W");
        int cnnLayerSize = hidden_linear_layers.size();
        for(int idx = cnnLayerSize - 1; idx >= 0; idx--)
            checkgrad.add(&hidden_linear_layers[idx].W, "hidden["+ std::to_string(idx) + "].W");
        checkgrad.add(&words.E, "words E");
    }

    // will add it later
    void saveModel(std::ofstream &os) const {
        wordAlpha.write(os);
        words.save(os);
        olayer_linear.save(os);
        labelAlpha.write(os);
    }

    void loadModel(std::ifstream &is) {
        wordAlpha.read(is);
        words.load(is, &wordAlpha);
        olayer_linear.load(is);
        labelAlpha.read(is);
    }

};

#endif /* SRC_ModelParams_H_ */