# How we define Machine Learning {#intro}

The demand for skilled data science practitioners in industry, academia, and government is rapidly growing. This book introduces concepts and skills that can help you develop a foundation that is missing in many educational platforms for Machine Learning. This book is a collection of my lecture notes that I have developed in the past 10 years.  Therefore, its language is not formal and it leaves most theoretical proofs to carefully selected cited sources.  It feels like a written transcript of lectures more than a reference textbook that covers from A to Z.

As of today (August 29, 2022), more than 103,942 people have been enrolled in the **Machine Learning** course offered online by Stanford University at [Coursera](https://www.coursera.org/learn/machine-learning).  The course is offered multiple times in a month and can be completed in approximately 61 hours.

I had a hard time for finding a good title for the book on a field where the level of interest is jaw-dropping.  Even finding a good definition for Machine Learning has became a subtle job as "machine learning" seems increasingly an *overloaded* term implying that a robot-like *machine* predicts things by learning itself without being explicitly programmed.  

Ethem Alpaydin, who is a professor of computer engineering, defines machine learning in the $3^{rd}$ edition of his book, [*Introduction to Machine Learning*](https://mitpress.mit.edu/books/introduction-machine-learning-third-edition) [@alpaydin_2014] as follows:

> Machine learning is programming computers to optimize a performance criterion using example data or past experience.  We have a model defined up to some parameters, and learning is the execution of a computer program to optimize the parameters of the model using the training data of past experience. (...) Machine learning uses the ***theory of statistics in building mathematical models***, because the core task is making inference from sample.  The role of computer science is twofold: First, in training, we need efficient algorithms to solve the optimization problem, as well as to store and process the massive amount of data we generally have.  Second, once the model is learned, its representation and algorithmic solution for inference needs to be efficient as well.
>

Hence, there are no "mysterious" machines that are learning and acting alone. Rather, there are well-defined **statistical models** for predictions that are optimized by efficient algorithms, and executed by powerful machines that we know as computers.  
