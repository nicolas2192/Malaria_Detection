import packages.Acquisition.terminal_cmd as ap
import packages.Acquisition.acquisition as aq
import packages.Wrangling.fix_data as wr
import packages.Model.model as ml
import packages.Reporting.reporting as rp

# Setting up constants.
DATAPATH = "data/raw"
CATEGORIES = ['Uninfected', 'Parasitized']
IMG_SIZE = 130


def main():
    # Argparse
    args = ap.terminal_parser()

    if args.train:
        # Loading the data
        data = aq.load_data(DATAPATH, CATEGORIES, IMG_SIZE)

        # Saving and loading data as a binary file. Uncomment these two lines to run it.
        # aq.save_binary(data, "../data/binary/data.pkl")
        # data_loaded = aq.load_binary("../data/binary/data.pkl")

        # Shuffling data to prevent over-fitting.
        shuffled = wr.shuffle_data(data)

        # Preparing data and splitting it into training set and label.
        X, y = wr.prep_data(shuffled)

        # Training the model
        model = ml.model_setup(X, y)

        # Saving the model as a h5 file
        ml.save_model_h5(model, model_path="data/model", model_name="model")

        # Evaluates a model performance
        ml.model_performance(model, X, y)

    if args.predict:
        # Loading the model from a h5 file
        model = ml.load_model_h5(model_path="data/model", model_name="model")

        # Predicting images
        image, result, image_path = ml.prediction(model, args.image, CATEGORIES, IMG_SIZE)

        # PDF generation if chose to do so
        if args.report:
            rp.pdf_generation(image, result, image_path)

    if args.train is False and args.predict is False:
        print('Type "python main.py -h" on your terminal to open the help menu')


if __name__ == "__main__":
    main()
