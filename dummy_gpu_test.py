import torch
import os



if __name__ == "__main__":

    print("Running torch.cuda.device_count()")
    print("  {:d}".format(torch.cuda.device_count()))
    print("Completed")

    #print("Running torch.cuda.is_available()")
    #print("  {:b}".format(torch.cuda.is_available()))
    #print("Completed")

    #if torch.cuda.is_available():
    if torch.cuda.device_count() > 0:

        print("Setting CUDA_DEVICE_ORDER environment variable")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Completed")

        print("Running device = torch.device(0)")
        device = torch.device(0)
        print("Completed")

        print("Running torch.cuda.empty_cache()")
        torch.cuda.empty_cache()
        print("Completed")

    print("Allocating (10) random tensor")
    tensor = torch.rand(10).to(device)
    print("  " + str(tensor))
    print("Completed")

    print("Done")
    