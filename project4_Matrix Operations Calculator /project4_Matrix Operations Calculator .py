import numpy as np
import sympy as sy
from random import randint

def generate_matrix():
    matrix1 = np.array([[randint(1,10) for _ in range(3)],[randint(1,10) for _ in range(3)],[randint(1,10) for _ in range(3)]])
    matrix2 = np.array([[randint(1,10) for _ in range(3)],[randint(1,10) for _ in range(3)],[randint(1,10) for _ in range(3)]])
    return matrix1,matrix2

def add(matrix1,matrix2):
    print("The first matrix is ")
    display(sy.Matrix(matrix1))
    print("The second matrix is ")
    display(sy.Matrix(matrix2))
    print(f"The addition is ")
    display(sy.Matrix(matrix1 + matrix2))
    

def sub(matrix1,matrix2):
    print("The first matrix = ")
    display(sy.Matrix(matrix1))
    print("The second matrix = ")
    display(sy.Matrix(matrix2))
    print(f"The subtraction is")
    display(sy.Matrix(matrix1 - matrix2))
    
    
def mul(matrix1,matrix2):
    print("The first matrix = ")
    display(sy.Matrix(matrix1))
    print("The second matrix = ")
    display(sy.Matrix(matrix2))
    print(f"The subtraction is ")
    display(sy.Matrix(np.dot(matrix1,matrix2)))
   

def scaler(matrix1,matrix2,scaler):
    print("The first matrix = ")
    display(sy.Matrix(matrix1))
    print("The second matrix = ")
    display(sy.Matrix(matrix2))
    print(f"The scaler multiplication of First Matrix is ")
    display(sy.Matrix(matrix1 * scaler))
    print(f"The scaler multiplication of Second Matrix is ")
    display(sy.Matrix(matrix2 * scaler))
    

def tran(matrix1,matrix2):
    print("The first matrix = ")
    display(sy.Matrix(matrix1))
    print("The second matrix = ")
    display(sy.Matrix(matrix2))
    print(f"The Transpose of matrix 1 is")
    display(sy.Matrix(matrix1.T))
    print(f"The Transpose of matrix 2 is")
    display(sy.Matrix(matrix2.T))

def main():
    while True:
        print("-"*20)
        print("Choose Your Option\n")
        print("1. Matrix Addition \n2. Matrix Subtraction \n3. Matrix Multiplication \n4. Scaler Multiplication \n5. Transpose of Matrix \n6. Exit")
        print("-"*20)

        user_input = int(input("Enter Your input (1-5)"))
        matrix1,matrix2 = generate_matrix()

        if not user_input:
            print("Please Enter something")
            
        else:           
            if user_input == 1:
                add(matrix1,matrix2)
                
            elif user_input == 2:
                sub(matrix1,matrix2)
                
            elif user_input == 3:
                mul(matrix1,matrix2)
    
            elif user_input == 4:
                scale = int(input("Enter the scaler that you want to multiply with"))
                scaler(matrix1,matrix2,scale)
    
            elif user_input == 5:
                tran(matrix1,matrix2)
    
            elif user_input == 6:
                print("You Exited")
                break
    
            else:
                print("Invalid Input Please try again")
                
if __name__ == "__main__":
    main()