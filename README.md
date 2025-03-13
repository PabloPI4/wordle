# wordle
Wordle is a game where the user have to guess a word using up to 6 attepmts. When a word is written in an attempt, the game will mark the letters of the word that are in the correct position in green, the letters that are in the word but in incorrect positions in yellow and the letters that are not in the word in grey. The game finishes when the word is guessed or the limit of attempts are reached.

This calculator is based on the idea of the information that every word gives for the next attempt. The fewer options a word leaves, the more information it provides.

For using the program, the user must execute it with the parameters: number of letters that contains the words (all the words must contain the same number of letters) and the file that contains the dictionary (each word in a different line).
Then the system will provide the 10 best options for the attempt. The user can select one of them or not, and then write the word selected, press enter and the sequence of colors that the wordle game returned (R grey, Y yellow, G green).
This will be executed until the number of attempts are equal to the limit or the user doesn't want the program to continue.

The cuda version is a parallelised version of the program, so it's time of execution is significantly lower.

dictionary.txt is an example of a valid and usable dictionary.
