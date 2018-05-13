# capstone-report
Project 7: Capstone Project : Predicting US High School Graduation Success
Author: jtmoogle @github.com   
Contanct info: jtmoogle@gmail.com for any questions or suggestions


README.md [README.md]( https://github.com/jtmoogle/capstone-report/blob/master/README.md)

- Folder location: [capstone-proposal]( https://github.com/jtmoogle/capstone-report)

Files are available for review
- (1)  [capstone_report.PDF]( https://github.com/jtmoogle/capstone-report/blob/master/capstone_report.pdf) capstone proposal
- (2)  [runme.pdf](https://github.com/jtmoogle/capstone-report/blob/master/runme.pdf)
- (3) Python source files [jtmoogle](https://github.com/jtmoogle/capstone-report/tree/master/jtmoogle)
   - [helper.py](https://github.com/jtmoogle/capstone-report/blob/master/jtmoogle/helper.py)
   - [hsgraduation.py](https://github.com/jtmoogle/capstone-report/blob/master/jtmoogle/hsgraduation.py)
   - [runme.py](https://github.com/jtmoogle/capstone-report/blob/master/jtmoogle/runme.py)
- (4) Data files https://github.com/jtmoogle/capstone-report/tree/master/jtmoogle/data
- Data File: [GRADUATION_WITH_CENSUS.csv](https://github.com/jtmoogle/capstone-report/blob/master/jtmoogle/data/GRADUATION_WITH_CENSUS.csv)
- Data Definition: [ALL_DATA_SCHEMA_M.pdf](https://github.com/jtmoogle/capstone-report/blob/master/jtmoogle/data/ALL_DATA_SCHEMA_M.pdf)

- (5)  [runme.ipynb ](https://github.com/jtmoogle/capstone-report/blob/master/runme.ipynb)

FYI, 
[proposal.pdf](https://github.com/jtmoogle/capstone-report/blob/master/proposal.pdf)
[proposal-review.pdf ]( https://github.com/jtmoogle/capstone-report/blob/master/proposal-review.pdf)
[proposal-studentnote.pdf ]( https://github.com/jtmoogle/capstone-report/blob/master/proposal-studentnote.pdf)


How-To Instruction:
- How to execute python source from local environment Python IDE (i.e Spyder 3.2.6)
   1. Manually load helper.py command: runfile('../capstone/jtmoogle/helper.py')
   2. Manually load hsgraduation.py command: runfile('../capstone/jtmoogle/hsgraduation.py')
   3. Manually load runme.py command: runfile('../capstone/jtmoogle/runme.py')
   4. Execute runme program command: runme(3,3)
Note: Output of analytic results were saved to .txt, .csv files. The plotting images were saved to .png files.
The capstone report pulls content directly from output files and images located at ../saved folder.

- How to execute runme.ipynb
   1. Manually launch Anaconda 3 Prompt
   2. type commands
         set mypath=c:\githup\capstone
         setfile=runme.ipynb
         activate capstone
         pip install --ignore-installed --upgrade tensorflow-gpu
         pip install ipykernel
         cd %mypath%
         python -c "from keras import backend"
         python -m ipykernel install --user --name capstone --display-name "Python (capstone)"
         python -c "import pandas"
         python -c "import jtmoogle.helper"
         python -c "import jtmoogle.hsgraduation"
         jupyter notebook %myfile%
   3. Expect browser to launch 'runme.ipynb' URL= http://localhost:8888/notebooks/runme.ipynb
   4. Manually click top menu "Cell" -> "Run All Below"

Note: Software versions
------------------------------
--> IPython version: 6.2.1
--> numpy version: 1.14.3
--> pandas version: 0.22.0
--> python version: 3.5.5
--> scikit-learn version: 0.19.1
--> sys version: 3.5.5 |Anaconda, Inc.| (default, Mar 12 2018, 17:44:09) [MSC v.1900 64 bit (AMD64)]
--> tensorflow version: 1.8.0
------------------------------


