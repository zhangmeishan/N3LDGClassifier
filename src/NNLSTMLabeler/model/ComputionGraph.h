#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder {
  public:
    const static int max_sentence_length = 1024;

  public:
    // node instances
    vector<LookupNode> _word_inputs;
    WindowBuilder _word_window;

    LSTM1Builder _lstm_left;
    LSTM1Builder _lstm_right;

    vector<ConcatNode> _lstm_concat;

    AvgPoolNode _avg_pooling;
    MaxPoolNode _max_pooling;
    MinPoolNode _min_pooling;

    ConcatNode _concat;

    LinearNode _neural_output;

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
        _word_window.resize(sent_length);
        _lstm_left.resize(sent_length);
        _lstm_right.resize(sent_length);

        _lstm_concat.resize(sent_length);

        _avg_pooling.setParam(sent_length);
        _max_pooling.setParam(sent_length);
        _min_pooling.setParam(sent_length);
    }

    inline void clear() {
        _word_inputs.clear();
        _word_window.clear();
        _lstm_left.clear();
        _lstm_right.clear();
    }

  public:
    inline void initial(ModelParams& model, HyperParams& opts) {
        for (int idx = 0; idx < _word_inputs.size(); idx++) {
            _word_inputs[idx].setParam(&model.words);
            _word_inputs[idx].init(opts.wordDim, opts.dropProb);
            _lstm_concat[idx].init(opts.hiddenSize * 2, -1);
        }
        _word_window.init(opts.wordDim, opts.wordContext);
        _lstm_left.init(&model.lstm_left_param, opts.dropProb, true);
        _lstm_right.init(&model.lstm_right_param, opts.dropProb, false);
        _avg_pooling.init(opts.hiddenSize * 2, -1);
        _max_pooling.init(opts.hiddenSize * 2, -1);
        _min_pooling.init(opts.hiddenSize * 2, -1);
        _concat.init(opts.hiddenSize * 3 * 2, -1);
        _neural_output.setParam(&model.olayer_linear);
        _neural_output.init(opts.labelSize, -1);
    }


  public:
    // some nodes may behave different during training and decode, for example, dropout
    inline void forward(Graph* pcg, const Feature& feature) {
        // second step: build graph
        //forward
        int words_num = feature.m_words.size();
        if (words_num > max_sentence_length)
            words_num = max_sentence_length;
        for (int i = 0; i < words_num; i++) {
            _word_inputs[i].forward(pcg, feature.m_words[i]);
        }
        _word_window.forward(pcg, getPNodes(_word_inputs, words_num));

        _lstm_left.forward(pcg, getPNodes(_word_window._outputs, words_num));
        _lstm_right.forward(pcg, getPNodes(_word_window._outputs, words_num));

        for (int i = 0; i < words_num; i++) {
            _lstm_concat[i].forward(pcg, &_lstm_left._hiddens[i], &_lstm_right._hiddens[i]);
        }

        _avg_pooling.forward(pcg, getPNodes(_lstm_concat, words_num));
        _max_pooling.forward(pcg, getPNodes(_lstm_concat, words_num));
        _min_pooling.forward(pcg, getPNodes(_lstm_concat, words_num));
        _concat.forward(pcg, &_avg_pooling, &_max_pooling, &_min_pooling);
        _neural_output.forward(pcg, &_concat);

    }
};

#endif /* SRC_ComputionGraph_H_ */