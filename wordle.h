//Macros of errors
#define ERROR_ARGUMENTS 1
#define ERROR_FILE_OPENING 2
#define ERROR_DYNAMIC_MEMORY_ALLOCATION 3
#define ERROR_LENGTH_WORD 4
#define ERROR_LENGTH_COLOR 5
#define ERROR_INVALID_COLOR 6
#define ERROR_NO_WORDS_POSSIBLE 7

//Types
typedef struct {
    int position;
    double entropy;
} entropyType;

//Functions
void wordle();
void calculateEntropies(entropyType *);
double calculateSingleEntropy(char *, char *, int);
int calculateProb(char *, char *);
int compEntropies(const void *, const void *);
void reduceDictionary(char *, char *);