import test_model as tm


def model_test():
    tm.run_test()
    print("Model is functional")
    print("Accuracy: " + str(tm.accuracy))
    
    if tm.accuracy > 0.5:
        print("Accuracy is greater than 0.5. Good job human!")

    if tm.accuracy < 0.9:
        print("You still have to gain " + str(0.99 - tm.accuracy) + " to reach 99%")


model_test()


