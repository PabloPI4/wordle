#include "wordle.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#define SIZE_WORDS 5

char *dictionary;
char *dictionaryGPU;
int tamDictionary;
int sizeWords;

__global__ void calculateEntropiesGPU(entropyType *, char *, int, int);
__device__ double calculateSingleEntropyGPU(char *, int, int, char *, char *, int);
__device__ int calculateProbGPU(char *, int, int, char *, char *);

/*
  The main function is responsible for checking that the number of parameters are correct, that the file can be 
  opened and for load the dictionary from the file
*/
int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "Length of the words and file name with all words of dictionary must be specified in the first argument\n");
        exit(ERROR_ARGUMENTS);
    }
    else if (argc > 3) {
        fprintf(stderr, "Incorrect number of arguments, it must be only two\n");
        exit(ERROR_ARGUMENTS);
    }

    FILE *file;
    if ((file = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "File \"%s\" cannot be opened\n", argv[2]);
        exit(ERROR_FILE_OPENING);
    }

    sizeWords = atoi(argv[1]);

    char line[sizeWords + 2];
    while(fgets(line, sizeWords + 2, file) != NULL) {
        if (tamDictionary % 64 == 0) {
            if ((dictionary = (char *) realloc(dictionary, sizeWords*(tamDictionary + 64))) == NULL) {
                fprintf(stderr, "Dynamic memory couldn't be allocated\n");
                exit(ERROR_DYNAMIC_MEMORY_ALLOCATION);
            }
        }

        for (int i = 0; i < sizeWords; i++) {
            if (line[i] == '\n') {
                fprintf(stderr, "Length of word \"%s\" at line %d does not match the specified length of the words\n", line, tamDictionary + 1);
                exit(ERROR_LENGTH_WORD);
            }

            dictionary[tamDictionary*sizeWords + i] = line[i];
        }

        tamDictionary++;
    }

    fclose(file);

    wordle();
}


/*
  The wordle function is the main function of the program, where an array of entropies are calculated and sorted to 
  give the 10 best options to the user
*/
void wordle() {
    int iteration = 1;

    //If iteration is 7 it means that the user lost the game
    while (iteration < 7) {
        //Entropies calculations and sorting
        entropyType entropies[tamDictionary];

        if (tamDictionary < 350) {
            calculateEntropies(entropies);
        }
        else {
            entropyType *entropiesGPU;

            cudaMalloc(&entropiesGPU, tamDictionary*sizeof(entropyType));
            cudaMalloc(&dictionaryGPU, tamDictionary*sizeWords);
            cudaMemcpy(dictionaryGPU, dictionary, tamDictionary*sizeWords, cudaMemcpyHostToDevice);

            int n_threads;
            int n_blocks;

            n_blocks = tamDictionary/768 + 1;
            if (tamDictionary < 768) {
                n_threads = tamDictionary;
            }
            else {
                n_threads = 768;
            }

            calculateEntropiesGPU<<<n_blocks, n_threads>>>(entropiesGPU, dictionaryGPU, tamDictionary, sizeWords);

            cudaMemcpy(entropies, entropiesGPU, tamDictionary*sizeof(entropyType), cudaMemcpyDeviceToHost);

            cudaFree(dictionaryGPU);
            cudaFree(entropiesGPU);
        }

        qsort(entropies, tamDictionary, sizeof(entropyType), compEntropies);

        //Giving the 10 best results to the user
        //If there isn't 10 options, it gives all the options
        for (int i = 0; i < 10; i++) {
            if (i == tamDictionary) {
                break;
            }
            char word[sizeWords + 1];
            strncpy(word, dictionary+entropies[i].position*sizeWords, sizeWords);
            word[sizeWords] = '\0';

            printf("%d: %s\n", i, word);
        }

        //Here the program ask the user which word has been selected and the result wordle gave
        char word[sizeWords];
        char colors[sizeWords];
        char useless;

        if (scanf("%c%c%c%c%c%c", word, word + 1, word + 2, word + 3, word + 4, &useless) < 6) {
            fprintf(stderr, "A word of length %d must be given\n", sizeWords);
            exit(ERROR_LENGTH_WORD);
        }
        if (scanf("%c%c%c%c%c%c", colors, colors + 1, colors + 2, colors + 3, colors + 4, &useless) < 6) {
            fprintf(stderr, "A color sequence of length %d must be given\n", sizeWords);
            exit(ERROR_LENGTH_COLOR);
        }

        for (int x = 0; x < sizeWords; x++) {
            if (colors[x] == 'g') {
                colors[x] = 'G';
                continue;
            }
            else if (colors[x] == 'y') {
                colors[x] = 'Y';
                continue;
            }
            else if (colors[x] == 'r') {
                colors[x] = 'R';
                continue;
            }
            if (colors[x] != 'G' && colors[x] != 'Y' && colors[x] != 'R') {
                fprintf(stderr, "Colors must be GYR\n");
                exit(ERROR_INVALID_COLOR);
            }
        }

        //Finally the dictionary is reduced following the word selected and color pattern obtained
        reduceDictionary(word, colors);

        if (tamDictionary == 0) {
            printf("No words are possible\n");
            exit(ERROR_NO_WORDS_POSSIBLE);
        }
    }
}


/*
  In this function all entropies are calculated
*/
void calculateEntropies(entropyType *entropies) {
    for (int i = 0; i < tamDictionary; i++) {
        char colors[sizeWords];

        entropies[i].position = i;
        entropies[i].entropy = -(calculateSingleEntropy(dictionary+i*sizeWords, colors, 0));
    }
}


/*
  Same function for gpu
*/
__global__ void calculateEntropiesGPU(entropyType *entropies, char *dictionary, int tamDictionary, int sizeWords) {
    char colors[SIZE_WORDS];

    int pos = blockIdx.x*blockDim.x + threadIdx.x;

    if(pos < tamDictionary) {
        entropies[pos].position = pos;
        entropies[pos].entropy = -(calculateSingleEntropyGPU(dictionary, tamDictionary, sizeWords, dictionary+pos*sizeWords, colors, 0));
    }
}


/*
  This is a recursive function that calculates the entropy of a word adding the information that a color pattern gives 
  to each other weighted
*/
double calculateSingleEntropy(char *word, char *colors, int depth) {
    double information = 0;

    if (depth == sizeWords - 1) {
        //In this case the color pattern is complete, so its information can be calculated

        colors[depth] = 'G';
        information += calculateProb(word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }
        
        colors[depth] = 'Y';
        information += calculateProb(word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }

        colors[depth] = 'R';
        information += calculateProb(word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }
    }
    else {
        //In this case the color pattern isn't complete, so its information is calculated in the function it calls

        colors[depth] = 'G';
        information += calculateSingleEntropy(word, colors, depth + 1);
        
        colors[depth] = 'Y';
        information += calculateSingleEntropy(word, colors, depth + 1);

        colors[depth] = 'R';
        information += calculateSingleEntropy(word, colors, depth + 1);
    }

    return information;
}


/*
  Same function for gpu
*/
__device__ double calculateSingleEntropyGPU(char *dictionary, int tamDictionary, int sizeWords, char *word, char *colors, int depth) {
    double information = 0;

    if (depth == sizeWords - 1) {
        //In this case the color pattern is complete, so its information can be calculated

        colors[depth] = 'G';
        information += calculateProbGPU(dictionary, tamDictionary, sizeWords, word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }
        
        colors[depth] = 'Y';
        information += calculateProbGPU(dictionary, tamDictionary, sizeWords, word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }

        colors[depth] = 'R';
        information += calculateProbGPU(dictionary, tamDictionary, sizeWords, word, colors)/tamDictionary;
        if (information != 0) {
            information *= log2(information);
        }
    }
    else {
        //In this case the color pattern isn't complete, so its information is calculated in the function it calls

        colors[depth] = 'G';
        information += calculateSingleEntropyGPU(dictionary, tamDictionary, sizeWords, word, colors, depth + 1);
        
        colors[depth] = 'Y';
        information += calculateSingleEntropyGPU(dictionary, tamDictionary, sizeWords, word, colors, depth + 1);

        colors[depth] = 'R';
        information += calculateSingleEntropyGPU(dictionary, tamDictionary, sizeWords, word, colors, depth + 1);
    }

    return information;
}


/*
  This function calculates the number of words that follows the pattern of the colors and word given
*/
int calculateProb(char *word, char *colors) {
    int numWords = 0;

    /*
      For each letter of each word of the dictionary
        if green is read in word given and letters don't match, the word cannot be a candidate
        if yellow is read in word given and letters match, the word cannot be a candidate
          if letters don't match then it's added to yellow letters list to see if it exists in another position
        if red is read in word given and letters match, the word cannot be a candidate
          if letters don't match then it's added to red letters list to check that it isn't exist in another position
    */
    for (int i = 0; i < tamDictionary; i++) {
        int numYellow = 0;
        char yellows[sizeWords];
        int numRed = 0;
        char red[sizeWords];
        int possible = 0;
        char validate[sizeWords];
        int pos;

        for (int j = 0; j < sizeWords; j++) {
            pos = i*sizeWords + j;

            for (int x = 0; x < numRed; x++) {
                if (red[x] == dictionary[pos]) {
                    break;
                }
            }

            if (colors[j] == 'G') {
                if (word[j] != dictionary[pos]) {
                    break;
                }
                else {
                    validate[j] = 'Y';
                }
            }
            else if (colors[j] == 'Y') {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    yellows[numYellow] = word[j];

                    numYellow++;
                    validate[j] = 'N';
                }
            }
            else {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    red[numRed] = word[j];

                    numRed++;
                    validate[j] = 'N';
                }
            }

            possible++;
        }

        if (possible < sizeWords) {
            continue;
        }

        possible = 1;

        int ny = numYellow;
        for (int x = 0; x < sizeWords; x++) {
            for (int y = 0; y < numYellow; y++) {
                if (yellows[y] == dictionary[i*sizeWords + x]) {
                    yellows[y] = 0;
                    ny--;
                    validate[x] = 'Y';
                    break;
                }
            }

            if (validate[x] == 'N') {
                for (int y = 0; y < numRed; y++) {
                    if (red[y] == dictionary[i*sizeWords + x]) {
                        possible = 0;
                        break;
                    }
                }
            }

            if (!possible) {
                break;
            }
        }

        if (!possible || ny > 0) {
            continue;
        }

        numWords++;
    }

    return numWords;
}


/*
  Same function for gpu
*/
__device__ int calculateProbGPU(char *dictionary, int tamDictionary, int sizeWords, char *word, char *colors) {
    int numWords = 0;
    char yellows[SIZE_WORDS];
    char red[SIZE_WORDS];
    char validate[SIZE_WORDS];

    /*
      For each letter of each word of the dictionary
        if green is read in word given and letters don't match, the word cannot be a candidate
        if yellow is read in word given and letters match, the word cannot be a candidate
          if letters don't match then it's added to yellow letters list to see if it exists in another position
        if red is read in word given and letters match, the word cannot be a candidate
          if letters don't match then it's added to red letters list to check that it isn't exist in another position
    */
    for (int i = 0; i < tamDictionary; i++) {
        int numYellow = 0;
        int numRed = 0;
        int possible = 0;
        int pos;

        for (int j = 0; j < sizeWords; j++) {
            pos = i*sizeWords + j;

            for (int x = 0; x < numRed; x++) {
                if (red[x] == dictionary[pos]) {
                    break;
                }
            }

            if (colors[j] == 'G') {
                if (word[j] != dictionary[pos]) {
                    break;
                }
                else {
                    validate[j] = 'Y';
                }
            }
            else if (colors[j] == 'Y') {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    yellows[numYellow] = word[j];

                    numYellow++;
                    validate[j] = 'N';
                }
            }
            else {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    red[numRed] = word[j];

                    numRed++;
                    validate[j] = 'N';
                }
            }

            possible++;
        }

        if (possible < sizeWords) {
            continue;
        }

        possible = 1;

        int ny = numYellow;
        for (int x = 0; x < sizeWords; x++) {
            for (int y = 0; y < numYellow; y++) {
                if (yellows[y] == dictionary[i*sizeWords + x]) {
                    yellows[y] = 0;
                    ny--;
                    validate[x] = 'Y';
                    break;
                }
            }

            if (validate[x] == 'N') {
                for (int y = 0; y < numRed; y++) {
                    if (red[y] == dictionary[i*sizeWords + x]) {
                        possible = 0;
                        break;
                    }
                }
            }

            if (!possible) {
                break;
            }
        }

        if (!possible || ny > 0) {
            continue;
        }

        numWords++;
    }

    return numWords;
}


/*
  In this function the dictionary is reduced to the words that follows the pattern of word and colors given
*/
void reduceDictionary(char *word, char *colors) {
    char *auxDict = NULL;
    int auxTamDict = 0;

    for (int i = 0; i < tamDictionary; i++) {
        int numYellow = 0;
        char yellows[sizeWords];
        int numRed = 0;
        char red[sizeWords];
        int possible = 0;
        char validate[sizeWords];
        int pos;

        for (int j = 0; j < sizeWords; j++) {
            pos = i*sizeWords + j;

            for (int x = 0; x < numRed; x++) {
                if (red[x] == dictionary[pos]) {
                    break;
                }
            }

            if (colors[j] == 'G') {
                if (word[j] != dictionary[pos]) {
                    break;
                }
                else {
                    validate[j] = 'Y';
                }
            }
            else if (colors[j] == 'Y') {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    yellows[numYellow] = word[j];

                    numYellow++;
                    validate[j] = 'N';
                }
            }
            else {
                if (word[j] == dictionary[pos]) {
                    break;
                }
                else {
                    red[numRed] = word[j];

                    numRed++;
                    validate[j] = 'N';
                }
            }

            possible++;
        }

        if (possible < sizeWords) {
            continue;
        }

        possible = 1;

        int ny = numYellow;
        for (int x = 0; x < sizeWords; x++) {
            for (int y = 0; y < numYellow; y++) {
                if (yellows[y] == dictionary[i*sizeWords + x]) {
                    yellows[y] = 0;
                    ny--;
                    validate[x] = 'Y';
                    break;
                }
            }

            if (validate[x] == 'N') {
                for (int y = 0; y < numRed; y++) {
                    if (red[y] == dictionary[i*sizeWords + x]) {
                        possible = 0;
                        break;
                    }
                }
            }

            if (!possible) {
                break;
            }
        }

        if (!possible || ny > 0) {
            continue;
        }

        if (auxTamDict % 32 == 0) {
            if ((auxDict = (char *) realloc(auxDict, (auxTamDict + 32)*sizeWords)) == NULL) {
                fprintf(stderr, "Dynamic memory couldn't be allocated\n");
                exit(ERROR_DYNAMIC_MEMORY_ALLOCATION);
            }
        }

        strncpy(auxDict + auxTamDict*sizeWords, dictionary + i*sizeWords, sizeWords);
        auxTamDict++;
    }

    if ((dictionary = (char *) malloc(sizeWords*auxTamDict)) == NULL) {
        fprintf(stderr, "Dynamic memory couldn't be allocated\n");
        exit(ERROR_DYNAMIC_MEMORY_ALLOCATION);
    }

    memcpy(dictionary, auxDict, sizeWords*auxTamDict);
    tamDictionary = auxTamDict;

    free(auxDict);
}


/*
  This function is needed for the qsort function call
*/
int compEntropies(const void *ent1, const void *ent2) {
    if (((entropyType *)(ent1))->entropy - ((entropyType *)(ent2))->entropy > 0) {
        return -1;
    }
    else {
        return 1;
    }
}