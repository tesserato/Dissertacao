---
title: 'Neural Networks - Overview and State of the Art'
author: 'Carlos Tarjano'
lang: 'en'
bibliography: bib.bib
csl: custom.csl
abstract: 'O crescimento exponencial do volume de dados gerados em nossa sociedade e a impossibilidade de tratá-los manualmente tem promovido cada vez mais a adoção de técnicas baseadas em machine learning nos mais variados contextos. Além disso, o aumento de poder capacional tem proporcionado a implementação de técnicas baseadas em inteligência artificial em cada vez mais dispositivos, como celulares.

No caso específico de redes neurais artificiais, observa-se uma multiplicação de termos e conceitos, na medida em que a base teórica e prática é extendida para novas aplicações, dificultando o acesso à area, já naturalmente fragmentada por seu caráter multidisciplinar e profundamente empírico / experimental.

Esses fatores comprometem uma visão holística da área, dificultando a migração de conhecimento entre subáreas, ou mesmo a importação e exportação de insights e comprometendo a sinergia e a identificação de questões de pesquisa relevantes, além de propiciar redundâncias de esforços de pesquisa na área.

Nesse contexto, o presente trabalho busca apresentar, a partir de uma extensiva revisão bibliográfica, uma visão macroscópica do atual estado da pesquisa e aplicação de redes neurais artificais, os principais autores, periódicos e campos de interesse, tendo como foco as áreas de aplicação onde essa ferramenta representa o estado-da-arte, além de identificar as principais questões em aberto na teoria pertinente.'
---

**keywords:** Neural Networks, Literature Review, Artificial Intelligence

# Motivação
Em face da ampla utilização de redes neurais nas mais diversas áreas, e de sua potencial utilidade para tantas outras, um trabalho identificando o estado da arte do campo em geral é bastante bem vindo.

Até onde vai nosso conhecimento, nenhum trabalho recente, com esse escopo geral, existe. Revisões em áreas específicas, algumas bastante abrangentes, servem a vários propósitos e serão aqui analisadas, mas não apresentam a visão sistêmica buscada neste trabalho.

Situando (potenciais) pesquisadores e informando-os dos problemas em aberto e das aplicações mais bem sucedidas da ferramenta, espera-se contribuir para a expansão do estudo e utilização de redes neurais artificiais, contribuindo para o progresso da área de inteligência artificial em geral.



# Estrutura
Na primeira parte do trabalho, como forma de uniformizar as diferentes terminologias e introduzir o tema, é apresentada uma rápida discussão sobre os principais conceitos e arquiteturas relacionados à redes neurais.

Em seguida são enumeradas as áreas onde redes neurais alcançam resultados de ponta

Posteriormente, são elencados os principais periódicos e congressos no setor, além dos autores mais proeminentes atualmente. Cabe ressaltar que trabalhos seminais, quando pertinente, serão citados nas bases teóricas.

O trabalho é concluído com os principais desafios e questões em aberto observadas.



# Escopo
Uma simples busca pelo termo neural networks em bases de dados científicas como Scopus e Web of Science revela centenas de milhares de resultados, tornando uma rígida definição do escopo de uma revisão bibliográfica sobre o tema imprescindível.

Em primeiro lugar, o artigo tem um foco prático, buscando delinear o estado da arte das redes neurais artificiais. Desta forma, trabalhos puramente teóricas sobre o tem serão evitados.

Excluindo os trabalhos seminais, apresentados na seção pertinente, serão priorizados artigos recentes.

# Metodologia
A pesquisa utiliza inicialmente a base de dados Scopus, de forma a delinear a grosso modo os principais campos de utilização de redes neurais artificiais.

A partir da análise dos resultados iniciais, ocorre um processo iterativo de identificação de temas, aplicações, autores e periódicos e novas consultas.

Pesquisas na base Web of Science e Google Scholar complementam os insights obtidos.



# Introdução



## Principais Arquiteturas
Being a mostly empyrical field,experimentation gave rise to a great variation in network types and topologies. We present below a non-exhaustive enumeration of the most prominent types of networks trying to establish, at the same time, a graduation of complexity and maintaining, when possible, a chronological order.

### Perceptron / Multilayer Perceptron

A Perceptron, one of the first steps towards the concrete use of neural networks, is a linear classifier model proposed in the late fifties by Frank Rosenblatt, capable of updating its weights to learn how to correctly classify linearly separable classes based on inputs and examples of desirable outputs.
It consists of an activation function f applied over the weighted sum of the inputs and the bias. Originally intended to be built as a machine for the Cornell Aeronautical Laboratory [@Rosenblatt57], its first implementation was in an IBM 704 computer simulation[@bishop06].

Multilayer Perceptrons are obtained fully connecting layers of Perceptrons.  Despite its apparent simplicity, this topology has been proven[@hornik91] to potentially work as an arbitrary precision function approximator. Given a sufficient number of hidden neurons, and arbitrary (non - polynomial[@leshno93]) activation functions, they can map any set of finite numbers to another[@hornik89]. Together with backpropagation and gradient descent, an efficient way of training networks proposed by Paul Werbos[@werbos74] in mid-seventies, they form the basis of the most common type of neural network, the feed-forward neural network, besides being used in many other architectures.   Below are two equivalent representations of a Perceptron; the left hand one illustrates its workings in more detail, while the other will be useful as a building block to the other architectures, presented in figure \ref{topologies}.

![Representation of the Perceptron](images/01perceptron.png)

### Feed-Forward Neural Networks
The principal characteristic of this network, as the name implies, is the forward motion of the inputs, without loops or cycles: Each signal flows from one fully connected layer to the next, until they reach the output nodes[@ojha2017metaheuristic]. Common activation functions are the sigmoidal, like the logistic
$y = \frac{1}{1 + e ^ {-x}}$   and the hyperbolic tangent $\tanh(x) = \frac{e^x − e^{-x}}{e^x + e^{-x}}$ [@goldberg2016primer]. They are a class encompassing single and multi layer Perceptrons.

### Deep Neural Networks
Deep neural networks are, in essence, feed-forward networks with many hidden layers, and thus, with a greater capacity of abstraction that, on the other hand, pose more difficulty to the training process. One of the main advantages of this type of topology is that the time-consuming task of extracting features from raw data is left to the networks themselves[@socher2014recursive]. Within each layer, more high-level representations can be created, based on the reasoning of the anterior layers. One drawback of this technique, however, is that it’s hard to infer exactly how this process is happening in the network.

### Recurrent
In recurrent networks the neurons are partially fed with their own states from past iterations, generating an effect that is similar to adding links from neurons to posterior neurons [@veit2016residual] in non-adjacent layers. This formulation, proposed by Elman [@elman1990finding] in 1990 with the aim of capturing features that could arise from serialisation in data is quite powerful in some applications, as information tagging [@xu2015ccg] for example.

### Convolutional
This type of (deep) feed-forward network is most commonly used in image applications, were they achieved state of the art performance in various tasks involving, for example, object and face recognition in images [@pang2017convolution]. Instead of receiving each pixel that forms an image as an input, a situation where fully connected layers can suffer rapidly from the curse of dimensionality, in this kind of network, the subsequent layers are sparsely connected with each other, conferring better trainability when compared with standard deep networks.

The convolutional layers can be interpreted as scanning the image [@lecun1998gradient], one stride at a time (generally 1 or 2 pixels), and storing the result of each step in a point of the subsequent layer. This allows for the detection of the relevant feature in different places of the image. To provide better generalisation, there are also layers that perform sub-sampling, simply reducing the dimensions of the anterior layer by averaging portions of them (or taking their maximum). After an arbitrary number of convolutional and sub-sampling layers, the actual reasoning is performed by a (series of) traditional, fully connected layer.

### Hopfield / Associative Memory Networks
Introduced by Hopfield [@hopfield1982neural] in 1982, this architecture where each neuron is connected to all others is among the most studied[@wu2012storage], and one of its foremost application is in the associative memory field, were, presented with an (imperfect) example of an object, the trained network can then retrieve its closest approximation.

![Topologies](images/02topologies.png)

## Principais Conceitos

### Support Vector Machines
While not strictly a type of neural network, this machine learning classification algorithm is closely related[@collobert2004links] to both the Perceptron and the multi-layer Perceptron.
Support Vector Machines, as proposed by Vapnik[@boser1992training] in 1982, are linear classifiers.

Replacing the dot product in the original algorithm with a kernel function allows nonlinear classification by means of transforming the original nonlinear space into a linearly separable space. This first proposition already covered its applicability to Perceptrons, thus albeit not a member of the neural network family, the use of the kernel function, commonly known as the kernel trick, is widely adopted in the field(e.g [@zhu2013kernel], [@huang2017learning], [@song2017robust]), and hence justifies the presentation of Support Vector Machines here.

### Extreme Learning Machines
Rather than a topology, this is an analytical method for training feed-forward neural networks with one hidden layer, proposed in 2004 [@huang2004extreme], with the aim of speeding up the traditional back propagation training algorithm.

### Generative Adversarial Networks
This technique/topology introduces the concept of one generative deep network creating content to be classified as artificially generated or not by another network [@goodfellow2014generative].

### Main Areas of Application


### Computer Vision
One of the most active research areas concerning the use of neural networks, this section encompasses image classification, synthesis and manipulation. To the best of our knowledge, the last broad review of the use of neural networks in this field dates back to 2002[@egmont2002image]. [@janai2017computer] presents a review of the state of the art in some tasks in this field that are related to autonomous vehicles, such as motion estimation, while proposing a taxonomy and review some pertinent datasets.

### Classification
This task consists of assigning a label to an image fed into the network.
The majority of image classification models make use of a deep convolutional topology, with added variations that include the substitution of the linear filters in the convolutional layers with shallow[@lin2013network] and deep[@pang2017convolution] feed-forward networks, the introduction of a max out pooling function[@liao2016importance], and the use of a category-selective objective function[@zhang2016improving].

[@sato2015apac] presents an augmented data technique in the context of data classification while [@wang2016sam] proposes, after an investigation of various topologies, an optimal architecture suited for the image classification task.

Also noteworthy is the work of [@hou2015blind], that interfaces image classification with sentiment analysis, proposing a non-quantitative methodology to assess image quality and the work of [@balle2015density], proposing and unsupervised methodology for data gaussianization.

Face and facial expression recognition can be defined as a subcategory of image classification, and thus literature reviewed in this area is presented here. The works of [@mollahosseini2016going] and [@li2015facial] present a deep network that assigns to a portrait the preponderance of one of six basic emotions, with almost 97% success rate in the latter.

With a new, generalizable architecture, [@zhao2016peak] proposes training optimised to the recognition of extreme facial expressions as a way of improving general expression recognition. Similarly, [@ding2017facenet2expnet] presents a two-stage training procedure to tackle the problem of small facial recognition datasets for the training of deep networks.

The existence of a reasonable number of labelled images datasets, like the MNIT, CIFAR and ImageNet databases, helps the training and development of image classification techniques. In the more specific scenario of medical image classification, however, no such datasets are readily available, and thus a smaller volume of research has been observed. [@shin2016deep] discusses this issue, proposing knowledge transfer techniques from general to medical datasets, while [@esteva2017dermatologist] employs some of the techniques suggested to elaborate a deep convolutional model, pre-trained in the ImageNet dataset to diagnose skin cancer, offering reliability comparable to human professionals.

### Object Detection
Object detection, both in images and videos, consists of identifying the presence of an arbitrary number of objects in the scene presented.

While the prominent architecture used in this task is also deep convolutional, [@bell2016inside] presents a technique that allows the identification of objects and contextual features to improve detection accuracy, while [@shi2016improving] advances the capability of baseline models by the introduction of min-max objective functions in later layers.

[@he2016deep] Achieved first place in ImageNet Large Scale Visual Recognition Challenge 2015  with the use of a residual network that enabled the efficient training of deeper architectures.

Considering the great number of datasets in which results are reported, and the lack of standardization, it's difficult to establish the absolute best results; nevertheless, state of the art, both in image classification and object recognition, seems to have been achieved with a wide residual network[@zagoruyko2016wide], that exchanges architecture deepness for broadness.

### Compression
Despite beeing an important field in face of the communication era demands[@rehman2014image], the last review about the use of neural networks in image compression was made in 1999[@jiang1999image], while the last general review about the area dates from 2014[@rehman2014image].

[@balle2016end] proposes the use of stages of linear convolutional filters and non-linear activation functions, improving the Multiscale Structural Similarity for Image Quality Assessment(MS-SSIM) measure in all bitrates.

Superior measurements, with the same metrics, to standard compression methods such as JPEG and WebP, were obtained by [@johnston2017improved], using a recurrent architecture with SSIM loss function and adaptive bit allocation algorithm.
With a modified loss function, [@theis2017lossy] shows that autoencoders can achieve compressing results comparable with JPEG 2000 format.

[@toderici2016full] also produces results better than standard codecs via different architectures based on recurrent neural networks, a binarizer, and a neural network for entropy coding.

Using a fuzzy neural network, [@wang2015image] achieves superior speed, robustness and quality in lossy image processing tasks. [@toderici2015variable] presents a progressive method, focused on reducing mobile phone data transfer, that allows arbitrary image quality depending on the quantity bits sent to the device. [@santurkar2017generative] investigates the resilience of neural networks based compression, via a generative model capable of offering graceful degradation on the compressed images. Conceptual compressing is investigated by [@gregor2016towards], a technique that allows images to be retrieved from symbols. All the literature investigate deals with lossy compression.

### Synthesis
[@isola2016image] presents a method to translate images via an adversarial network, generating images from outlines, for example, or providing automatic image colorization [@hwangimage], [@zhang2016colorful], [@larsson2016learning]  and  [@iizuka2016let], the last focusing on automatic image colorization via a convolutional network, based on the extraction of local and global features, learned in the supervised training process.

Also using adversarial nets, [@frans2017outline] introduces control to the process of generating coloured images from sketches, with the use of colour maps fed into the , a concept that is further explored in [@sangkloy2016scribbler], where user interaction and a feed-forward architecture enables real-time colorization of images via the input of colour clues via scribbles in arbitrary areas of the image.

[@gatys2016image] tackles the problem of style transfer: extracting features of an image and applying to another one, without changing the semantics of the latter, with the application of a convolutional network.

[@kulkarni2015deep], on the other hand, proposes a method that enables a convolutional-deconvolutional network to disentangle the features extracted from images, allowing the manual generation of images in different positions and lighting conditions via the tweak of variables fed to the network. Also investigating image manipulation, by means of adversarial nets, [@zhu2016generative] attempts to learn features directly from raw data.

[@oord2016pixel] proposes a deep recurrent topology, with improved residual connections, capable of reconstructing occluded images, that could be also used in image compression tasks.

[@theis2015generative] investigates a recurrent architecture composed of multi-dimensional long short term memory units in the context of modelling image distributions.

### Video
Problems in this area can be understood as a general case of aforementioned image-related tasks, with the added complexity of taking advantage of temporal correlations and a much higher data dimensionality.  Motivated by this, [@karpathy2014large] investigates approaches capable of extending convolutional neural networks in order to enable them to take advantage of temporal information in the inputted data. In a similar attempt, extending facial expression recognition to videos, [@khorrami2016deep] merges convolutional and recurrent networks, measuring the relative relevance of each one in the final results. Tackling the curse of dimensionality, [@Yang2017] presents the tensor-train concept, enabling the transfer of improvements of other architectures to high dimensional sequential data. [@he2015multimodal] uses a deep bidirectional long short-term memory recurrent neural net to set benchmarks in the recognition of emotions, via audio and video processing.

### Language
[@deng2013new] offers an overview of the field, as of 2013, based on the papers submitted to a special session at ICASSP-2013. A more recent short survey is presented by [@goldberg2016primer] with the aim of acquainting newcomers to the field, fomenting the migration of sparse liner models to dense neural network approaches.

### Speech
An year before, [@hinton2012deep] present an overview of the use of neural networks based approach in the field, while in this same year, [@graves2013speech] produces state of the art results in the TIMIT phoneme recognition benchmark using a deep recurrent neural network, while [@maas2013rectifier] points the superiority of rectifier nonlinearities over sigmodal activation functions in the task of continuous speech recognition.

With a hybrid architecture, combining a recurrent phonetic model and a deep neural network acoustic classifier, [@boulanger2014phone]  applies phone sequencing strategies to set new benchmarks in the TIMIT dataset, a technique that is proved by [@sak2015fast] to be superior to architectures like deep long short-Term memory recurrent neural networks and the widely used hidden Markov models.

[@sainath2015deep] investigates the optimization of convolutional nets hyperparameters, pooling and training strategies to applications in speech recognition tasks. [@zweig2017advances] and [@zhang2017towards]  investigates end-to-end systems, with the latter combining hierarchical convolutional nets with Connectionist Temporal Classification. In his work, [@zhang2017very] also explores end-to-end systems, via a very deep recurrent convolutional network employing NIN principles.

Speech synthesis state of the art, as seen in text-to-speech applications, for instance, is not yet achieved via end-to-end neural network approaches. Nonetheless, this area is being actively researched, and approaching production quality rapidly. [@zen2015unidirectional] tackles this task using unidirectional long short-term memory recurrent neural networks with a recurrent output layer, while [@wu2016investigating] further investigates this architecture, trying to discover the reasons for its effectiveness, and pinpoint wich factor are more relevant to que quality of the task, with the goal of offering a simplified topology.

[@sotelo2017char2wav], [@wang2017tacotron] and [@arik2017deep] present an end-to-end solution; the first via a two-component model, composed of a bidirectional recurrent neural network as an encoder,  and a recurrent net as the decoder. The second approach involves a single neural net, fed with textual data and speech samples during training. The third is based on the traditional speech synthesis pipeline,  substituting all the 6 components with neural networks.

### text
[@kim2014convolutional] investigates the performance of convolutional nets in the task of text classification and sentiment analysis, while [@xu2015ccg] introduces the uses of recurrent topologies in the task, reporting performance improvements of about 1% in the results generated.

Appling a bidirectional long short-term memory to extract expressions and a convolutional net to classify sentences [@chen2017improving] stablishes new benchmarks in the area. [@xu2017deep] presents a similar approach, achieved in a single pass of a convolutional net, while [@araque2017enhancing] experiments with ensembles techniques.

### Music
Many language processing techniques are used in the music field; similarly, many image tasks can be translated to music, given a suitable representation for the input data, like spectrograms.

### Classification
This task involves assigning tags, generally genre-related or emotion related, to musical pieces. [@costa2017evaluation] and [@choi2016automatic] tackles this task with a fully convolutional neural network fed, in the latter case, with music represented by a mel-spectrogram, while in [@choi2017convolutional] a recurrent architecture is also explored, to exploit the temporal correlation of the inputs.

### Transcription
the common task in this field is to translate music parts, generally specific instruments, into a symbolic representation, like tablatures or music scores, for example. One of the first contributions to this field is seen in [@tuohy2006evolved] via the coupling of a network and a local heuristic hill-climber applied over the results, to generate tablatures from music.

With the use of a recurrent net, [@boulanger2013high] transcribes spectrograms of general musical parts into piano roll midi commands, while [@bock2012polyphonic] offers a similar approach, restricted to polyphonic piano sound, as in the case of [@sigtia2016end]. The last work, however, uses different architectures for the acoustic, a simple network and the language, a recurrent network model. With the use of a bidirectional recurrent net fed with spectral representations, [@southall2016automatic], creates drum representations.

### Generation
Here the aim is the inverse of that in the music transcription: given a representation, audio output is generated. One of the first works in the field is seen in[@stanley2007compositional], consisting of a compositional pattern producing network generating music on the fly based on user input.  [@hutchings2017talking] generates full drum parts based on a kick drum pattern, with a recurrent net, and investigates the quality of the results via an online survey.

### Forecasting
Some work has been developed in this area, in which the main topology used is the recurrent net, like in [@zhu2013kernel]. In this paper kernel modifications are also applied, in line with later work present by[@song2017robust] in which a training algorithm is introduced, improving performance and generalisation capabilities.

An important economical application of forecasting techniques concerns the prediction of stock prices,  where [@wang2016financial] introduces a random recurrent net approach. [@di2016artificial] investigates the fitness of multiple architectures in the task, from feed-forward to convolutional nets.

[@barrow2016cross] mixes k-fold and MonteCarlo cross-validation with neural networks, reporting improvements in both short and long term predictions.
[@bas2016training] proposes an evolutive algorithm, based on the multiplicative neuron model[@yadav2007time], capable of generating optimal hyperparameters for the forecasting task.
[@tu2017mapping] utilises the NeuCube spiking neural network architecture, originally designed with medical applications[@kasabov2012neucube], to the forecasting task, via the introduction of new training and variable mapping algorithms.

### Architecture Optimisation
Much research effort has been made in the task of determining optimal hyperparameters for neural networks, with the use of various methods. Evolutionary algorithms are the prominent approach here, as can be seen in [@fritzke1994growing], [@yao1999evolving], two of the first works on the field, that already points the potential synergy between evolutionary algorithms and neural networks. One of the more prominent evolutive algorithms is proposed in [@stanley2002evolving], and serves as a baseline measure to various posterior works.

A supervised, probabilistic approach is formulated by [@mao2000probabilistic], while the genetic approaches are once more investigated in [@leung2003tuning] and in [@ritchie2003optimizationof], that concludes that, for the task of mapping genes associated with common diseases, an evolved network performs better than a manually tuned one.

As an alternative to backpropagation's computational cost,[@palmes2005mutation] suggests a mutation-based genetic neural network.

[@tsai2006tuning] once more tackles the problem from an evolutive standpoint, making use of a hybrid Taguchi-genetic algorithm, improving over results reported in the literature, while [@ludermir2006optimization] formulates a simulated annealing methodology, reporting improvement over other optimisation procedures and [@benardos2007optimizing] presents a genetic algorithm specific for the optimization of feed-forward nets.

An ant colony technique is present by [@salama2015learning] while an adaptative technique based on a solid theoretical analysis with generalisation guarantees is presented in [@cortes2016adanet]

The concept of neural fabrics, embedding multiple potential architectures, is introduced [@saxena2016convolutional], reducing to two the number of hyperparameters to be tuned. [@floreano2008neuroevolution] prepared a review in 2008, comparing different neuroevolutional methods.

More recently, [@ojha2017metaheuristic] surveys more than 300 papers, investigating the use of conventional metaheuristics-baed optimisation procedures, limited in scope to feed-forward nets. Restricting his work to applications related to games, [@risi2015neuroevolution] offers a review of the state of the art of neuroevolutive techniques.

More specific optimisation investigations are conducted both by [@lee2016generalizing] in the context of pooling functions to improve convolutional nets and [@park2016analysis] analysing dropout techniques.

### Conclusion
Summarize the major contributions,
evaluating the current position, and
pointing out flaws in methodology, gaps
in the research, contradictions and areas
for further study


# References
