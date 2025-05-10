---
title: "Foundations of Statistical and Machine Learning for Actuaries"
author:
  - Edward (Jed) Frees, University of Wisconsin - Madison
  - Andrés Villegas Ramirez, University of New South Wales
date: today
format:
  html:
#  docx:
    include-before-body: ShowHideShortCourse.js
    number-sections: true
    encoding: "UTF-8"
    date-format: "D MMMM YYYY"
output-dir: docs
editor: source
---






::: {.cell layout-align="center"}

:::






**Course Overview.** This short course introduces statistical and machine learning with an emphasis on actuarial applications. Our approach stems from the fact that many modern machine learning tools can be interpreted through the lense of statistical principles, including classical regression techniques. Beginning with an overview of the statistical foundations, we show how to develop models based on "learning" from the data. We emphasize classical techniques widely used in actuarial applications such as those based on generalized linear models. This development of the foundation naturally leads to a modern approach towards statistical analysis known as statistical learning.
For this approach, we describe techniques such as those based on boosting and tree-based methods, including random forests.

<h6 style="text-align: left;"><a id="displayText.Begin.Hide" href="javascript:toggleText('toggleText.Begin.Hide','displayText.Begin.Hide');"><i><strong>More on the Course Overview: Statistical and Machine Learning.</strong></i></a> </h6><div id="toggleText.Begin.Hide" style="display: none">  In contrast to statistical learning, machine learning only uses algorithms that can learn from the data and do not employ an underpinning probabilistic framework. Due to their popularity and widespread use, we provide an overview with an emphasis on actuarial applications. Our main focus is on neural networks and the "deep learning" associated with many layers of a network. Without the need to develop reasoning based on probabilistic models, the machine learning community has developed approaches to handle a broad set of problems including those from computer vision, text recognition, and natural language processing; we demonstrate these applications in this course. 

This course will provide participants with a guided tour of the statistical and machine learning landscapes. Our goals are to: 

(1)  give participants a deep understanding of what this set of tools can achieve in actuarial contexts, and 
(2)  provide a springboard for participants that will allow them to develop additional expertise using machine learning approaches.

***

</div>

<br>

**Course Format.** The format of the course will consist of alternating blocks between presentations of the underlying principles and practical applications. In a typical block, the instructor will spend 45 minutes reviewing the insurance motivation and key mathematical underpinnings. This will be followed by a 45 minute block of time in which participants will actively explore a selected case study. Thus, it is anticipated that participants will bring a laptop.

<h6 style="text-align: left;"><a id="displayText.Begin2.Hide" href="javascript:toggleText('toggleText.Begin2.Hide','displayText.Begin2.Hide');"><i><strong>More on the Course Format: Coding in R and Python.</strong></i></a> </h6><div id="toggleText.Begin2.Hide" style="display: none"> For the foundations, the statistical learning component, and many of the case studies, we will utilize the statistical package 'R'. This package is written by statisticians and its approach is tied directly to statistical applications. Participants with some familiarity with 'R' will benefit most from the course. For the machine learning components, we provide code from the general programming language 'python'. We do not assume prior knowledge of python but will provide code so that participants can calibrate machine learning models using case studies. One of the benefits of the course is to expose participants to the power of python.

***

</div>

<br>

**Target Audience**: Practicing actuaries, students, and educators interested in exposure to the foundations of insurance analytics.

# Short Course Brief Outline {-}

-  Day 1 - Foundations and Statistical Learning
-  Day 2 - Statistical and Machine Learning
   -  Lecture on Ethical Aspects of AI from Dr. Fei Huang
-  Day 3 - More on Machine Learning and How it Affects Actuarial Practice 
   -  Lecture on AI applications in insurance context from Prof. Dani Bauer
   

   
<h6 style="text-align: left;"><a id="displayText.Section4122.Hide" href="javascript:toggleText('toggleText.Section4122.Hide','displayText.Section4122.Hide');"><i><strong>A More Detailed Plan</strong></i></a> </h6><div id="toggleText.Section4122.Hide" style="display: none">   

*  Let us assume that each two hour block consists of 50 minutes of lecture, with 40 minutes interactive work and a 30 minute break.
*  So, four blocks per day, with three days in total





::: {.cell layout-align="center"}
`````{=html}
<table class=" lightable-classic table table-striped table-condensed" style='font-family: "Arial Narrow", "Source Sans Pro", sans-serif; width: auto !important; margin-left: auto; margin-right: auto; margin-left: auto; margin-right: auto;'>
<caption><b>Detailed Schedule</b></caption>
 <thead>
  <tr>
   <th style="text-align:center;font-weight: bold;"> Day and  <br> Time </th>
   <th style="text-align:center;font-weight: bold;"> Presenter </th>
   <th style="text-align:left;font-weight: bold;"> Topics </th>
   <th style="text-align:center;font-weight: bold;"> Presenter - Data Source </th>
   <th style="text-align:center;font-weight: bold;"> Participant Activity - Data Source </th>
  </tr>
 </thead>
<tbody>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Monday <br> Morning </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Welcome and Foundations <br> Hello to Google Colab </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction">Seattle House Prices</a><br>French Motor Liability Data </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://openacttextdev.github.io/RegressionSpanish/index.html">Health Expenditures</a> </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; ">  </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Logistic Regression and Generalized Linear Models </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://openacttextdev.github.io/RegressionSpanish/index.html">Health Expenditures</a><br>French Motor Liability Data </td>
   <td style="text-align:center;width: 8.5cm; "> LogisticGLM.ipynb <br><a href="https://openacttextdev.github.io/RegressionSpanish/index.html">Health Expenditures</a> </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Monday <br> Afternoon </td>
   <td style="text-align:center;width: 1cm; "> Andrés </td>
   <td style="text-align:left;width: 8.5cm; "> Regularization, Resampling, Cross-Validation </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction">Seattle House Prices</a> </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction">Seattle House Prices</a> </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; ">  </td>
   <td style="text-align:center;width: 1cm; "> Andrés </td>
   <td style="text-align:left;width: 8.5cm; "> Classification </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://discover.data.vic.gov.au/dataset/victoria-road-crash-data">Victoria road crash data</a> </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://discover.data.vic.gov.au/dataset/victoria-road-crash-data">Victoria road crash data</a> </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Tuesday <br> Morning </td>
   <td style="text-align:center;width: 1cm; "> Andrés </td>
   <td style="text-align:left;width: 8.5cm; "> Trees, Boosting, Bagging </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; ">  </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Big Data, Non-Supervised Learning, Dimension Reduction </td>
   <td style="text-align:center;width: 8.5cm; "> MNIST Data </td>
   <td style="text-align:center;width: 8.5cm; "> MNIST Data </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Tuesday <br> Afternoon </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Artificial Neural Networks </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction">Seattle House Prices</a><br>French Motor Liability Data </td>
   <td style="text-align:center;width: 8.5cm; "> <a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction">Seattle House Prices</a> </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; ">  </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Graphic Data Neural Networks </td>
   <td style="text-align:center;width: 8.5cm; "> MNIST Data </td>
   <td style="text-align:center;width: 8.5cm; "> MNIST Data </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Tuesday <br> 4 pm </td>
   <td style="text-align:center;width: 1cm; "> Fei </td>
   <td style="text-align:left;width: 8.5cm; "> Ethics - Fei Huang </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; "> None </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Wednesday <br> Morning </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Recurrent Neural Networks </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; ">  </td>
   <td style="text-align:center;width: 1cm; "> Jed </td>
   <td style="text-align:left;width: 8.5cm; "> Artificial Intelligence and ChatGPT </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; "> None </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Wednesday <br> After Lunch </td>
   <td style="text-align:center;width: 1cm; "> Dani </td>
   <td style="text-align:left;width: 8.5cm; "> Dani Bauer Insights </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; "> None </td>
  </tr>
  <tr>
   <td style="text-align:center;width: 1.0cm; "> Wednesday <br> Afternoon </td>
   <td style="text-align:center;width: 1cm; "> Andrés </td>
   <td style="text-align:left;width: 8.5cm; "> Applications and Wrap-Up </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
   <td style="text-align:center;width: 8.5cm; ">  </td>
  </tr>
</tbody>
</table>

`````
:::



<br>


-  Day 1 - *Foundations and Statistical Learning*
   - For the foundations, we can draw from Jed's Bogotá short course as a base
   - For statistical learning, we can draw from the course of Andrés given in Africa as a base. We also have the James et al book for this. In addition, we have Peng Shi and UNSW courses for sample code and examples.
-  Day 2 - *Statistical and Machine Learning*   
   - For machine learning, we can rely upon the Géron (2023) book. We also have the Dani Bauer and UNSW courses for sample code and examples.
-  Day 3 - *More on Machine Learning and How it Affects Actuarial Practice* 
   - Use the IFoA: Actuaries Analytical Cookbook, Actuarial Data Science tutorials, and Peng Shi course to develop a structure. (Actually, we have too many publicly available examples.)
   
</div>


# Google Colaboratory and Jupyter Notebooks {-}

To deliver this course, we will utilize two resources that will likely be unfamiliar to some participants.

*  [Google colaboratory](https://colab.research.google.com/) (colab for short) is a cloud-based system of servers designed to process machine learning code.
   *  We will use the free base package - you only need a (free) Google account.
   *  Colab handles both R and python code - perfect for our needs.
   *  Machine learning applications often depend upon large datasets and utilize computationally intensive algorithms - Colab is designed to accomodate these demands.
*  [Jupyter notebooks](https://nbviewer.org/github/OpenActTextDev/ActuarialRegression/blob/main/Notebooks/Introduction%20to%20Jupyter%20Notebooks.ipynb) provide a handy way to combine executable code, code outputs, and text into one connected file.    
   *  You can take a look at the course notebooks by going to [nbviewer site](https://nbviewer.org/). Then, enter the course Github account URL [https://github.com/OpenActTextDev/ActuarialRegression](https://github.com/OpenActTextDev/ActuarialRegression/tree/main/Notebooks), select the folder and then a notebook that you want to view.
   *  To interact with notebook, go to Colab!


# Data {-}

For this short course, you will find links to the data embedded in our notebooks (that you will retrieve on the fly). So, you will not need to download data in advance for this course.

However, should you be interested in experimenting further, here are some helpful sources of data:

-   [Short Course Repo](https://github.com/OpenActTextDev/ActuarialRegression/tree/main/CSVData) - this is the Github repository where we will store data for the short course.
    -   This repo contains data from the [Modelado de Regresión con Aplicaciones Actuariales y Financieras](https://openacttextdev.github.io/RegressionSpanish/index.html). You may also download these datasets from the [Frees Regression Book Data Site](https://instruction.bus.wisc.edu/jfrees/jfreesbooks/Regression%20Modeling/BookWebDec2010/data.html). Here, you will also find additional documentation regarding variable definitions.
-   [Loss Data Analytics List of Other Data Resources](https://openacttexts.github.io/LDAVer2/DataResources.html#other-data-sources) - in this open actuarial textbook, many interesting datasets are described, including the Kaggle competitions datasets.
-   [Predictive Modeling in Actuarial Science](https://instruction.bus.wisc.edu/jfrees/jfreesbooks/PredictiveModelingVol1/index.htm) These are some other data sets, with R code, useful for regression applications.



# Short Course Resources

## Other actuarial courses

-   Madison
    -   [Frees Bogotá Foundations Short Course](https://openacttextdev.github.io/RegressionSpanish/ShortCourseOrganization.html)
    -   [Frees Regression Book in Spanish](https://openacttextdev.github.io/RegressionSpanish/index.html)
    -   [Dani Bauer's Teaching Github Repo](https://github.com/danielbauer1979/MSDIA_PredictiveModelingAndMachineLearning)
    -   [Dani Bauer's CAS Tutorial 2022 on Github](https://github.com/danielbauer1979/CAS_PredMod)
-   Australia
    -   [Villegas Ramirez 2023 AFRIC Statistical learning workshop, Github Repo](https://github.com/amvillegas/afric)
    -   [UNSW; Statistical Machine Learning for Risk and Actuarial Applications](https://unsw-risk-and-actuarial-studies.github.io/ACTL3142/). This course relies heavily on James et al.; see below.
    -   [UNSW: Artificial Intelligence and Deep Learning Models for Actuarial Applications](https://laub.au/DeepLearningForActuaries/)
        -   [Course GitHub Resources](https://github.com/Pat-Laub/DeepLearningForActuaries)
        -   [Course Motivation](https://www.actuaries.digital/2023/05/18/teaching-ai-to-unsw-actuarial-students/)
    -   [UNSW Actuarial Software Directory](https://unsw-risk-and-actuarial-studies.github.io/)        
-  Switzerland
    -   [Github Repo For Actuarial Deep Learning Short Course](https://github.com/actuarial-data-science/CourseDeepLearningWithActuarialApplications)
-   Other
    -   [Machine Learning for Actuarial Science - Github Repo](https://github.com/xiangshiyin/machine-learning-for-actuarial-science)
    -   This course and UNSW course rely heavily on [Géron, A. (2022). Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow (3rd ed.). O'Reilly Media.](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/).
        -  [Free Github Code for Géron](https://github.com/ageron/handson-ml3)
 

## Other actuarial resources

-  UK
   -   [IFoA: Actuaries Analytical Cookbook](https://actuariesinstitute.github.io/cookbook/docs/index.html)
   -   [Getting Started in Machine Learning](https://institute-and-faculty-of-actuaries.github.io/mlr-blog/workstreams/foundations/)
-  Switzerland
   -   [Actuarial Data Science](https://actuarialdatascience.org/)
   -   [Github Repo](https://github.com/actuarial-data-science/Tutorials)  
   -    [AI Tools for Actuaries (March 3 2025), Mario Wüthrich et al.](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5162304)   
   -   [Statistical Foundations of Actuarial Learning and its Applications](https://link-springer-com.virtual.anu.edu.au/book/10.1007/978-3-031-12409-9)
-  Rest of Europe
   -   [CPD in Data Science](https://actuary.eu/wp-content/uploads/2024/09/2024-09-16_CPD-in-DS-FINAL.pdf)
   -   Effective Statistical Learning Methods for Actuaries (2019/20), by Michel Denuit, Donatien Hainaut , Julien Trufin (**not free**) 
       -  [Part 1: GLMs and Extensions](https://link.springer.com/book/10.1007/978-3-030-25820-7)
       -  [Part II: Tree-Based Methods and Extensions](https://link.springer.com/book/10.1007/978-3-030-57556-4)
       -  [Part III: Neural Networks and Extensions](https://www.amazon.com/Effective-Statistical-Learning-Methods-Actuaries-ebook/dp/B07ZVK2H43)
 
## Thought Provoking Articles and Commentaries 

-   [Recent Challenges in Actuarial Science](https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-040120-030244), by Embrechts and Wütrich
-   [What an actuary should know about artificial intelligence](https://actuary.eu/wp-content/uploads/2024/01/What-should-an-actuary-know-about-Artificial-Intelligence.pdf)
-   [Wolfram Overview of ML](https://www.wolfram.com/language/introduction-machine-learning/)
    -   [What Is ChatGPT Doing - and Why Does It Work?](https://writings.stephenwolfram.com/2023/02/what-is-chatgpt-doing-and-why-does-it-work/) 
   



## Other Courses (Some with Books)

### Statistical Learning

-   [James et al. (2021), An Introduction to Statistical Learning with Applications in R](https://www.statlearning.com/)
-   [James et al. (2023), An Introduction to Statistical Learning with Applications in Python](https://www.statlearning.com/)
    -   [Stanford Statistical Learning](https://www.edx.org/learn/python/stanford-university-statistical-learning-with-python)
    -   [YouTube](https://www.youtube.com/watch?v=yLEx1FnYyOo&list=PLoROMvodv4rNHU1-iPeDRH-J0cL-CrIda&index=11)

### Machine Learning and AI


-   [Coursera - Machine Learning by Andrew Ng](https://www.coursera.org/learn/machine-learning)
-   [Lecun ML Course](https://atcold.github.io/NYU-DLSP21/)
-   [Build a Large Language Model (From Scratch)](https://lightning.ai/courses/deep-learning-fundamentals/)
    -   [Raschka Teaching Site](https://sebastianraschka.com/teaching/)
-   [Python Machine Learning](https://archive.org/details/python-machine-learning-and-deep-learning-with-python-scikit-learn-and-tensorflow-2)
    -   [Github Repo](https://github.com/rasbt/python-machine-learning-book-3rd-edition)
-   [Fast AI](https://course.fast.ai/)


## Books

-   [ Chollet, F. (2021). Deep learning with Python. Simon and Schuster.](https://sourestdeeds.github.io/pdf/Deep%20Learning%20with%20Python.pdf)
    -  [Python Github Repo](https://github.com/fchollet/deep-learning-with-python-notebooks)
    -  [R Github Repo](https://github.com/t-kalinowski/deep-learning-with-R-2nd-edition-code)
    -  [A Friendly R Intro to Keras](https://keras3.posit.co/index.html)
    -  [Datacamp Course Overview](https://www.datacamp.com/tutorial/keras-r-deep-learning)
-   [Introduction to Neural Network Models of Cognition](https://com-cog-book.github.io/com-cog-book/intro.html)
-   [Deep Learning - Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- *Book:* ["Speech and Language Processing" by Dan Jurafsky and James H. Martin](https://web.stanford.edu/~jurafsky/slp3/)
     - *Course:* [Stanford CS224n - Natural Language Processing with Deep Learning - YouTube Lectures](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)
-   [Hands-on Large Language Models](https://jalammar.github.io/illustrated-transformer/)
-   [Grokking Deep Learning](https://www.manning.com/books/grokking-deep-learning)
    -    [Github Repo](https://github.com/iamtrask/Grokking-Deep-Learning)
-   [An interesting collection of free books](https://github.com/yanshengjia/ml-road/tree/master)


# Learn Python
 
In this course, we will **not** assume knowledge of Python although we will assume that participants have some familiarity with R. Through exposure to the Python scripts, we simply want to demonstrate how machine learning tools can be utilized.

Our hope is that the course will inspire some attendees to continue learning about machine learning approaches after the course. You will find that the Python machine learning community to be much larger than the corresponding R users community. If you decide to continue on a machine learning journey, you probably will want to get additional exposure to Python. Here are a few selected resources:

-   [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)
    -  [Beyond the Basic Stuff with Python](https://inventwithpython.com/beyond/) - Some great explanations
-   [UNSW: Artificial Intelligence and Deep Learning Models for Actuarial Applications](https://laub.au/DeepLearningForActuaries/)
-   [Numpy](https://numpy.org/doc/stable/user/absolute_beginners.html)
-   [IFoA: Actuaries Analytical Cookbook](https://actuariesinstitute.github.io/cookbook/docs/index.html)

<br>

**Some Ways that Python differs from R**

-   Purpose
    -  R: Designed for statisticians by statisticians.
    -  Python: General-purpose language (web dev, automation, AI, etc.) with growing data science capabilities.
-   Indexing starts at 0, not 1
-   There are methods (functions) associated with data types. Each data type has its own set of methods.
-   Python using indentation rules for blocks of code! (How old fashion... Like cobol.) (The Python standard is 4 spaces.)
-   You can use "1_000" for "1000", the underscore improves readability....





