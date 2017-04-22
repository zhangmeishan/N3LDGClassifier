#ifndef SRC_ComputionGraph_H_
#define SRC_ComputionGraph_H_

#include "ModelParams.h"


// Each model consists of two parts, building neural graph and defining output losses.
struct GraphBuilder{
public:
	const static int max_sentence_length = 1024;

public:
	// node instances
	vector<LookupNode> _word_inputs;
	WindowBuilder _word_window;
	vector<UniNode> _hidden;

	AvgPoolNode _avg_pooling;
	MaxPoolNode _max_pooling;
	MinPoolNode _min_pooling;

	ConcatNode _concat;

	LinearNode _neural_output;

  Graph *_pcg;

public:
	GraphBuilder(){
	}

	~GraphBuilder(){
		clear();
	}

public:
	//allocate enough nodes 
	inline void createNodes(int sent_length){
		_word_inputs.resize(sent_length);
		_word_window.resize(sent_length);
		_hidden.resize(sent_length);
	}

	inline void clear(){
		_word_inputs.clear();
		_word_window.clear();
		_hidden.clear();
	}

public:
	inline void initial(Graph* pcg, ModelParams& model, HyperParams& opts, AlignedMemoryPool* mem = NULL){
    _pcg = pcg;
		for (int idx = 0; idx < _word_inputs.size(); idx++) {
			_word_inputs[idx].setParam(&model.words);
			_word_inputs[idx].init(opts.wordDim, mem);
			_hidden[idx].setParam(&model.hidden_linear);
			_hidden[idx].init(opts.hiddenSize, mem);
		}
		_word_window.init(opts.wordDim, opts.wordContext, mem);
		_avg_pooling.init(opts.hiddenSize, mem);
		_max_pooling.init(opts.hiddenSize, mem);
		_min_pooling.init(opts.hiddenSize, mem);
		_concat.init(opts.hiddenSize * 3, mem);
		_neural_output.setParam(&model.olayer_linear);
		_neural_output.init(opts.labelSize,mem);
	}


public:
	// some nodes may behave different during training and decode, for example, dropout
	inline void forward(const Feature& feature, bool bTrain = false){
    _pcg->train = bTrain;
		// second step: build graph
		//forward
		int words_num = feature.m_words.size();
		if (words_num > max_sentence_length)
			words_num = max_sentence_length;
		for (int i = 0; i < words_num; i++) {
			_word_inputs[i].forward(_pcg, feature.m_words[i]);
		}
		_word_window.forward(_pcg, getPNodes(_word_inputs, words_num));

		for (int i = 0; i < words_num; i++) {
			_hidden[i].forward(_pcg, &_word_window._outputs[i]);
		}
		_avg_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		_max_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		_min_pooling.forward(_pcg, getPNodes(_hidden, words_num));
		_concat.forward(_pcg, &_avg_pooling, &_max_pooling, &_min_pooling);
		_neural_output.forward(_pcg, &_concat);
		
	}
};

#endif /* SRC_ComputionGraph_H_ */