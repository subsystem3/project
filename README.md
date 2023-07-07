# Final Team Project

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/team2-su23-aai501/project/tree/dev?quickstart=1)

## Introduction

The final team project in AAI 501 will give you an opportunity to identify an AI-driven problem, perform a hands-on project, and deliver a report and presentation as a team. To deliver a successful project, you should carefully read the final team project criteria and timelines.

## Team Structure

Complete the final team project survey by Friday (Day 4) of the first week/module. Your instructor will assign you to a group with 2 or 3 members by the end of Week 1.

## AI and Machine Learning Problems and Datasets

For this teamwork project, you may choose your own AI research problem and dataset. It can be a problem or a challenge from your workplace or one found in an online data source, such as [Kaggle](https://www.kaggle.com/datasets) or [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml). A list of potential project ideas is provided below for your reference, but feel free to choose any AI challenges not in this list that excites you the most:

- The application of AI techniques, such as machine learning and deep learning algorithms, to the healthcare and medical field, has gained a lot of traction in the last few years. There are medical datasets such as Digital Database for Screening Mammography (DDSM) with a low signal-to-noise ratio that is available through open source projects such as Cancer Imaging Archive project. Make informed decisions in determining which classification algorithms you learned from this class can be used to detect disease and also offer some interpretation of your model's findings.
- Can we predict human activity from smartphone sensor data? Classify the readings from multiple sensors in a smartphone to identify activities like walking, walking upstairs, walking downstairs, sitting, standing, and lying. UCI has produced a [labeled dataset](http://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) related to this task.
- Understanding gender bias in book reviews using the techniques and data from [this project](https://md.ekstrandom.net/pubs/book-author-gender), along with [the code](https://github.com/BoiseState/bookdata-tools).
- Image classification is another alternative project. Use a pre-trained network or train your own (or some combination of both!) to classify images into specific categories. Some interesting datasets to use for this problem include the following:
  - The [notMNIST dataset](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html) expands on the classic MNIST dataset of handwritten digits: These are digits rendered in stylistic forms that you might see in logos as opposed to handwritten digits. The goal is to predict the digit from the image.
  - Classify the images in the scientific literature: The [Viziometrics](http://viziometrics.org/api/) dataset includes some hand-labeled images extracted from PubMed and other sources.

- You may also consider a more straightforward classification problem as a project, such as those from competition sites such as [Kaggle](https://www.kaggle.com/) or other open datasets. However, if you take this option, apply your creative spin on these projects rather than just "training a classifier." These projects are often unrealistically "clean" for a practical project. In particular, for structured datasets with 10-20 features, you will want to explain what the model learned, not just report accuracy. For example, explore variable importance to understand which features were most important in making good predictions. Then you may want to compare results with simpler models that use only these important features. If those simpler models work well, that is an interesting, actionable result—we can get adequate results by collecting just a few key features. Examples are:
  - Predict the loan status [fully paid, late, etc.] of a Lending Club loan, given the borrower's information.
  - Predict whether a transaction is fraudulent, given anonymized credit card transactions.
  - Use NYSE data and fundamental metrics to predict things like one-day ahead prediction, the quality of momentum strategies, or security clustering strategies.
  - Use data on a breast cancer tumor to predict whether it is benign or malignant.
  - Build a network to predict who will die in the next season of Game of Thrones.

Some rules/tips about choosing AI challenges and data sets for your final projects:

1. Do not choose the problems that we have already analyzed in the course.
1. The dataset should not be small or made up. For this course, "small" is defined as fewer than 1000 examples in the dataset.
1. Choose a data set that does not require excessive data preprocessing.

## Experiment Design

Define a problem on the dataset and describe it in terms of its real-world organizational or business application. The complexity level of the problem should be at least comparable to one of your assignments. The problem may use at least two different types of AI and machine learning algorithms that we have studied in this course, such as Classification, Clustering, and Regression, in an investigation of the analytics solution to the problem. This investigation must include some aspects of experimental comparison. Depending on the problem, you may choose to experiment with different types of algorithms, e.g., different types of classifiers and some experiments with tuning parameters of the algorithms. Alternatively, if your problem is suitable, you may use multiple algorithms (Clustering + Classification, etc.). Note that if there are a larger number of attributes in your selected dataset, you can try some type of feature selection to reduce the number of attributes. You may use summary statistics and visualization techniques to help you explain your findings in this project.

## Proposal

To ensure that you choose an appropriate project, turn in a 1-2 page proposal by the end of week 3. Use this proposal document to demonstrate that you have completed some background work on your chosen topic. The proposal should begin with a clear and unambiguous statement of your topic and include all of the following:
A brief discussion of the problem and algorithms you intend to investigate and the system you intend to build in doing so.
Identification of specific related course topics (e.g., heuristic search, classification, deep learning, NLP, CV, etc.).
Examples of expected behaviors of the system or the types of problems the algorithms you investigate are intended to handle.
The issues you expect to focus on.
A list of papers/articles or other resources you intend to use to inform your project efforts. This list forms the core of your project report reference list in APA 7.
Make sure that you will work very closely and constantly communicate with all of your teammates throughout the project in delivering multiple project deliverables.

## Final Project Presentation

Prepare and record a final team project presentation by the end of Week 7. You may use any recording software you wish. Ensure that the sound quality of your video is good and that each member presents an equal portion of the presentation. The final project presentation is a chance to explain your problem and approach, provide your analysis results and interpretation, showcase what you have accomplished, and discuss your next step plan.

Upload your final team project presentation slides and group presentation, which should be between 20-30 minutes. Note that EVERY group member must participate in the final team project presentation. You need to upload the recording to a video-sharing website, such as YouTube.com or Vimeo.com, and share the link to the recording on the title page of your presentation slides.

## Final Project Paper and Code

You must submit a well-written report on your final project and the complete, well-documented, and clean source code by the end of Week 7. Your report (without Appendices), including text and selected tables/graphs, should be 5-10 pages in length and describes the AI algorithms you implemented or deployed together with the data on which they were tested. Furthermore, you should include a detailed analysis of results or system performance (depending on the project's scope). We strongly recommend GitHub as a method of collaborating and submitting source code. Write and submit your final project report in APA 7 style similar to (sample professional paper).

The report should contain the following contents:

- A description of the purpose, goals, and scope of your project. You should include references to papers you read on which your project and any algorithms you used are based. Include a discussion of whether you adapted a published algorithm or devised a new one, the range of problems and issues you addressed, and the relation of these problems and issues to the techniques and ideas covered in this course.
- A clear specification of the AI algorithms you used with analysis, evaluation, and critique of the algorithm and your implementation. For algorithm comparison, it is preferred that empirical comparison results are presented graphically.
- An appendix that provides a list of each project participant and their detailed contributions to the project.
- Your code should be clearly documented. Submit your code with the project report to Blackboard (or link directly to a public repository in your report). Remember that your project report serves as the tour guide for your readers to be able to repeat your project process and discover the same patterns as you did. Only one member of your team will need to submit the deliverables during this final project.

## Project Timeline

- Module 1 (by the end of week 1): The course instructor will group students into teams of two to three members. Blackboard, USD Email, or Slack may be used to contact team members.
- Module 3 (by the end of week 3): One team representative will need to submit your 1-2 page project proposal.
- Module 7 (by the end of week 7): One team representative should submit deliverables for the course project in the final week:
- Final Project Presentation Slides (including the video link)
- Final Project Paper and Code

It is critical to note that no extensions will be given for any of the final team project due dates for any reason, and final team projects submitted after the final due date will not be graded.

**Plagiarism, or passing another person's work/code off as one's own either by directly copying or even paraphrasing it without proper citation, is a serious offense and can result in sanctions, including grade reductions, course failures, and even expulsion from the university. For more information, please see the USD Code of Honor.

For details on how this assignment will be assessed, view the  assignment rubric.

Note: You will submit the Peer Evaluation form individually using the separate assignment link in Module 7. Consult the syllabus for grading weights of the final team project and peer evaluations.
