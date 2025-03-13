# Lab 2 Code

To execute the new code I implemented, use either of the following commands:

For building a random skeleton:
```bash
python3 rigid_body.py 4 train
```

For building a circular skeleton:
```bash
python3 rigid_body.py 5 train
```

You can also change the user parameters for either function call in `rigid_body.py` near the bottom of the file inside the `main()` function.

Most of the implementation details are in my lab report but generally, I chose to use for loops to generate all the bodies and joints, using randomization for the random skeleton generation, and certain formulas for the circular generation. Both are meant to be attempts at generating different shapes within a general class (either random or wheel-like) while allowing the user to choose certain parameters. This is opposed to the existing robot examples where evertyhing is hard coded.