#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder {
  public:
    const static int max_sentence_length = 100;

  public:
    // node instances
    vector<LookupNode> _word_inputs;

    LSTM1Builder _lstm_left;
    LSTM1Builder _lstm_right;

    vector<ConcatNode> _lstm_concat;


    MaxPoolNode _max_pooling;

    LinearNode _neural_output;

    Graph *_pcg;

  public:
    GraphBuilder() {
    }

    ~GraphBuilder() {
        clear();
    }

  public:
    //allocate enough nodes
    inline void createNodes(int sent_length) {
        _word_inputs.resize(sent_length);

        _lstm_left.resize(sent_length);
        _lstm_right.resize(sent_length);
        _lstm_concat.resize(sent_length);


    }

    inline void clear() {
        _word_inputs.clear();
        _lstm_left.clear();
        _lstm_right.clear();
        _lstm_concat.clear();
    }

  public:
    inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts) {
        _pcg = pcg;
        for (int idx = 0; idx < _word_inputs.size(); idx++) {
            _word_inputs[idx].setParam(&model.words);
            _word_inputs[idx].init(opts.wordDim, -1);
            _lstm_concat[idx].init(opts.hiddenSize * 2, -1);
        }

        _lstm_left.init(&model.lstm_left_params, -1, true);
        _lstm_right.init(&model.lstm_right_params, -1, false);

        _max_pooling.init(opts.hiddenSize * 2, -1);

        _neural_output.setParam(&model.olayer_linear);
        _neural_output.init(opts.labelSize, -1);
    }


  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(const Feature& feature, bool bTrain = false) {
        _pcg->train = bTrain;
        // second step: build graph
        //forward
        int words_num = feature.m_words.size();
        if (words_num > max_sentence_length)
            words_num = max_sentence_length;
        for (int i = 0; i < words_num; i++) {
            _word_inputs[i].forward(_pcg, feature.m_words[i]);
        }

        _lstm_left.forward(_pcg, getPNodes(_word_inputs, words_num));
        _lstm_right.forward(_pcg, getPNodes(_word_inputs, words_num));
        for(int i = 0; i < words_num; i++) {
            _lstm_concat[i].forward(_pcg, &_lstm_left._hiddens[i], &_lstm_right._hiddens[i]);
        }

        _max_pooling.forward(_pcg, getPNodes(_lstm_concat, words_num));

        _neural_output.forward(_pcg, &_max_pooling);
    }
};

#endif /* SRC_ComputionGraph_H_ */
