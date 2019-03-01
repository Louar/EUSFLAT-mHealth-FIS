# Takagi-Sugeno Fuzzy Inference System to 'understand' personal preferences of mHealth users

This repository is made publicaly available to foster reprodution of the results of the article *Application of fuzzy modelling to 'understand' personal preferences of mHealth users: a case study* that is submitted to [EUSFLAT 2019](http://eusflat2019.cz). In this case study, it is evaluated whether personal preferences can be understood from user event data in an mHealth setting. Based on a theoretical framework, user preferences are described using six parameters, i.e., classes. Based on this framework, a structure of six -- one for each class -- Takagi-Sugeno fuzzy inference systems is constructed and evaluated. In this case study, it is found that user preferences may be derived from user event data using fuzzy modeling with accuracy scores that are higher then a random predictor would typically achieve.



## Getting Started
1. Clone the **master** branch from this GitHub repository, e.g. using [Sourcetree](https://www.sourcetreeapp.com)
2. Import the local repository into and IDE of your preference, e.g. [Atom](https://atom.io)
3. Run the file `TsFisController.py`


### Project structure
```
/
├── datasets/
├── output-final/
│   ├── performance/
│   ├── rules/
│   ├── out-validation.txt
│   └── out.txt
├── output/
│   ├── performance/
│   └── rules/
├── README.md
├── TsFisController.py
└── TsFisService.py
```


## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct that applies, and the process for submitting pull requests.

## Authors
* **Raoul Nuijten** - *Corresponding author* - [personal website](http://www.projectraoul.nl)

## License
This project is licensed under the ... License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments
This work is part of the research program ‘[Gamification for Overweight Prevention and Active Lifestyle](https://healthgoal.eu)’ (443001101), which is partly financed by the Netherlands Organisation for Health Research and Development ([ZonMw](https://www.zonmw.nl/en/)).
