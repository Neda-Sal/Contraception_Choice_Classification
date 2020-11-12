
# Code Review: Project 3

[X] README.md included <br/>
[X] Slides Included <br/>
[X] Code Included <br/>

## Clean Code:
* Generally follows PEP-8
* I think it's inefficient to duplicate/drop rows for mapping, you might as well just copy the whole df at once if you want to preserve the original
* Effective use of discrete modules 

## Good Documentation:
* Well organized readme
* Could stand to include additional/more detailed markdown cells for analysis and explanation
* Good use of docstrings where appropriate


## Proper Data Science:
* Comprehensive investigation of candidates models and data representation
* Shap is cool!
* Consider exploring hyperparameter tuning after settling on your final model(s)

## Comments:
* Great topic choice and readme 
* Don't forget SQL has a "between" condition that can save you some typing
* df.nunique can be applied to the whole df at once
* It's fine to include personality in your documentation, but I'd suggest being mindful of professionalism if you plan to showcase your work (e.g. "preggers" kind of stands out)
