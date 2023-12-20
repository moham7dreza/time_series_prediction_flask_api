import matplotlib.pyplot as plt


class ResultPlotter:
    @staticmethod
    def plotting(dataset, name):
        # Plotting
        plt.figure(figsize=(12, 8))

        # Plotting open, high, low, close prices
        plt.plot(dataset.index, dataset['<OPEN>'], label='Open', linestyle='-', color='blue')
        plt.plot(dataset.index, dataset['<HIGH>'], label='High', linestyle='-', color='green')
        plt.plot(dataset.index, dataset['<LOW>'], label='Low', linestyle='-', color='red')
        plt.plot(dataset.index, dataset['<CLOSE>'], label='Close', linestyle='-', color='purple')

        # Adding grid
        plt.grid(True, linestyle='--', alpha=0.6)

        # Adding labels and title
        plt.xlabel('Date(EN)')
        plt.ylabel('Price(Rial)')
        plt.title('<' + name + '> Prices Over Time')

        # Formatting x-axis with a readable date format
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        # Rotating x-axis labels for better readability
        plt.gcf().autofmt_xdate()

        # Adding legend with a shadow
        plt.legend(shadow=True)

        # Adding a background color to the plot
        plt.gca().set_facecolor('#f4f4f4')

        # Adding horizontal lines for reference
        plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Reference Line')

        # Display the plot
        plt.show()

    @staticmethod
    def plot_result(x, h, title):
        plt.bar(x, h)
        plt.ylabel("Price")
        plt.title("Average Stock Price Predictions for " + title)
        plt.show()

    @staticmethod
    def result_plotting(x, h, title):
        # Plotting
        plt.figure(figsize=(12, 8))

        # Plotting open, high, low, close prices
        plt.bar(x, h, label=title, linestyle='-', color='blue')

        # Adding grid
        plt.grid(True, linestyle='--', alpha=0.6)

        # Adding labels and title
        plt.xlabel('Models')
        plt.ylabel('Price(Rial)')
        plt.title('<' + title + '> Prices Over Time')

        # Formatting x-axis with a readable date format
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.gca().xaxis.set_major_locator(mdates.DayLocator())

        # Rotating x-axis labels for better readability
        plt.gcf().autofmt_xdate()

        # Adding legend with a shadow
        plt.legend(shadow=True)

        # Adding a background color to the plot
        plt.gca().set_facecolor('#f4f4f4')

        # Adding horizontal lines for reference
        plt.axhline(y=100, color='gray', linestyle='--', linewidth=1, label='Reference Line')

        # Display the plot
        plt.show()
