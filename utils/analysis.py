import pandas as pd
from matplotlib import pyplot as plt
from utils.plot_stacked_percentage_bar_chart import plot_stacked_percentage_bar_chart

def analysis(df_train):

    # B  - Business: Travel related to business or work purposes.
    # HBE - Home-Based Education: Travel from home to educational institutions (like school or university) or back home.
    # HBO - Home-Based Other: Travel originating from home for other activities (e.g., appointments, errands).
    # HBW - Home-Based Work: Commuting from home to work or from work back to home.
    # NHBO - Non-Home-Based Other: Travel not originating from home for non-work or non-education-related purposes, such as leisure or social activities.

    # Adjust layout to avoid clipping
    plt.tight_layout()

    # Plot travel_mode over day_of_week
    plot_stacked_percentage_bar_chart(df_train, feature='day_of_week', target='travel_mode')

    # Plot purpose over day_of_week
    plot_stacked_percentage_bar_chart(df_train, feature='day_of_week', target='purpose')

    # Plot travel_mode over purpose
    plot_stacked_percentage_bar_chart(df_train, feature='travel_mode', target='purpose')

    # Plot Age Density of Study Participants
    df_train['age'].plot(kind='kde', figsize=(10, 6))
    plt.title('Age Density Plot of Study Participants')
    plt.xlabel('Age')
    plt.ylabel('Density')

    # Plot age_binned over purpose
    age_bins = range(0, 101, 10)  # Creates bins [0, 10), [10, 20), ..., [90, 100]
    df_train['age_binned'] = pd.cut(df_train['age'], bins=age_bins, right=False)
    plot_stacked_percentage_bar_chart(df_train, feature='age_binned', target='purpose')

    # Plot age_binned over travel_mode
    plot_stacked_percentage_bar_chart(df_train, feature='age_binned', target='travel_mode')

    # Identify participants in the age group 20-30 who have traveled for education
    df_train['unique_id'] = df_train['household_id'].astype(str) + '_' + df_train['person_n'].astype(str)
    edu_participants = df_train[(df_train['age_binned'].astype(str) == '[20, 30)') & (df_train['purpose'] == 'HBE')]['unique_id']

    # Create a new column 'student_status'
    df_train['student_status'] = 'Non-Student'
    df_train.loc[(df_train['unique_id'].isin(edu_participants)), 'student_status'] = 'Student'

    # Plot student_status over travel_mode
    plot_stacked_percentage_bar_chart(df_train[df_train['age_binned'].astype(str) == '[20, 30)'], 'student_status', 'travel_mode')

    # Plot travel_month over travel_mode
    plot_stacked_percentage_bar_chart(df_train, feature='travel_month', target='travel_mode')

    # Plot trip count over travel_month
    ax = df_train['travel_month'].plot.hist(alpha=1, bins=12, figsize=(10, 6), edgecolor='white' )
    # Set axis labels
    ax.set_xlabel('Travel Month')
    ax.set_ylabel('Number of Trips')
    # Set title
    ax.set_title('Trip Count of Each Month')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Plot distance_binned over purpose
    df_train['distance_binned'] = pd.cut(df_train['distance'], bins=10, right=False)
    plot_stacked_percentage_bar_chart(df_train, feature='distance_binned', target='purpose')

    # Plot age_binned over travel_mode
    plot_stacked_percentage_bar_chart(df_train, feature='distance_binned', target='travel_mode')

    # Plot cost_transit_binned over travel_mode
    df_train['cost_transit_binned'] = pd.cut(df_train['cost_transit'], bins=10, right=False)
    plot_stacked_percentage_bar_chart(df_train, feature='cost_transit_binned', target='travel_mode')

    # Plot cost_transit_binned over distance
    plt.scatter(df_train['cost_transit'], df_train['distance'], alpha=0.5, c='blue', edgecolors='w', s=60)
    # Add title and labels
    plt.title('Transit Cost vs Distance', fontsize=16)
    plt.xlabel('Transit Cost', fontsize=14)
    plt.ylabel('Distance', fontsize=14)
    # Show grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Plot travel_mode stacked
    travel_mode_counts = df_train['travel_mode'].value_counts()
    # Create a list of colors for each travel mode
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    # Plotting a single stacked bar chart
    plt.figure(figsize=(8, 6))
    # Initialize the bottom position for stacking
    bottom = 0
    # Plot each segment of the bar
    for i, (mode, count) in enumerate(travel_mode_counts.items()):
        plt.bar(
            x=['Travel Modes'],  # Only one bar
            height=count,
            color=colors[i],
            width=0.6,
            bottom=bottom,
            label=mode
        )
        # Update the bottom position for the next segment
        bottom += count
    plt.ylabel('Counts')
    plt.title('Distribution of Travel Modes (Stacked Bar)')
    plt.xticks([])
    plt.legend(title='Travel Mode')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
