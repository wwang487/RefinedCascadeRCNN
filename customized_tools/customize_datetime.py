from datetime import datetime, timedelta
from astral.sun import sun
from astral import LocationInfo
import pytz

import os

from tqdm import tqdm

def format_time(time_str):
    """Convert a string in HH-MM format to a datetime.time object."""
    return datetime.strptime(time_str, "%H-%M").time()

def find_missing_timestamps(base_dir, start_time='08-00', end_time='18-00', interval=15):
    start_time = format_time(start_time)
    end_time = format_time(end_time)
    expected_images_per_hour = 3600 // interval
    missing_timestamps = {}  # Dictionary to store missing timestamps

    # List directories to estimate total progress accurately
    directories = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    for folder_name in tqdm(directories, desc="Processing folders"):
        folder_path = os.path.join(base_dir, folder_name)
        all_images = [f for f in os.listdir(folder_path) if f.endswith('-PW.jpg')]
        image_times = set(datetime.strptime(img[:-7], '%Y-%m-%d-%H-%M-%S') for img in all_images)

        # Start and end datetime for the day
        day_date = datetime.strptime(folder_name, '%Y-%m-%d')
        start_datetime = datetime.combine(day_date, start_time)
        end_datetime = datetime.combine(day_date, end_time)

        # Prepare timestamps to progress over
        all_timestamps = []
        current_time = start_datetime
        while current_time < end_datetime:
            all_timestamps.append(current_time)
            current_time += timedelta(hours=1)

        missing_day = []  # List to collect missing timestamps for the day

        for current_time in tqdm(all_timestamps, desc=f"Checking {folder_name}", leave=False):
            hour_end = min(current_time + timedelta(hours=1), end_datetime)
            # Count images within this hour
            image_count = sum(1 for img_time in image_times if current_time <= img_time < hour_end)
            if image_count < expected_images_per_hour:
                # If fewer images than expected, check each timestamp
                while current_time < hour_end:
                    next_time = current_time + timedelta(seconds=interval)
                    if not any(current_time <= img_time < next_time for img_time in image_times):
                        missing_day.append(current_time.strftime('%Y-%m-%d-%H-%M-%S'))
                    current_time = next_time
            else:
                # Skip detailed checking if the hour is fully covered
                current_time = hour_end

        missing_timestamps[folder_name] = missing_day

    return missing_timestamps

def sort_datetime_list(datetime_list):
    """
    Sorts a list of datetime strings from the earliest to the latest time.

    Parameters:
        datetime_list (list of str): List of datetime strings in the format "yyyy-mm-dd-hh-mm-ss".

    Returns:
        list of str: A sorted list of datetime strings.
    """
    # Convert strings to datetime objects and sort
    sorted_datetimes = sorted(datetime_list, key=lambda x: datetime.strptime(x, "%Y-%m-%d-%H-%M-%S"))

    # Convert datetime objects back to strings if needed (optional step if you need strings)
    # sorted_datetime_strings = [dt.strftime("%Y-%m-%d-%H-%M-%S") for dt in sorted_datetimes]

    return sorted_datetimes


def filter_dict_by_datetime_range(data_dict, start_datetime, end_datetime):
    """
    Filters a dictionary by datetime range.

    Parameters:
        data_dict (dict): Dictionary with datetime strings as keys.
        start_datetime (str): Start datetime as a string "yyyy-mm-dd-hh-mm-ss".
        end_datetime (str): End datetime as a string "yyyy-mm-dd-hh-mm-ss".

    Returns:
        dict: A dictionary containing key-value pairs within the specified datetime range.
    """
    # Convert start and end datetime strings to datetime objects
    start = datetime.strptime(start_datetime, "%Y-%m-%d-%H-%M-%S")
    end = datetime.strptime(end_datetime, "%Y-%m-%d-%H-%M-%S")

    # Filter the dictionary
    filtered_dict = {key: value for key, value in data_dict.items()
                     if start <= datetime.strptime(key, "%Y-%m-%d-%H-%M-%S") <= end}

    return filtered_dict

def get_next_date(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    next_day = date_obj + timedelta(days=1)
    next_day_str = next_day.strftime('%Y-%m-%d')
    return next_day_str

def is_endtime_reached(date_str1, date_str2):
    """
    Check if the first date is ahead of the second date.

    Args:
    date_str1 (str): First date string in the format 'YYYY-MM-DD'.
    date_str2 (str): Second date string in the format 'YYYY-MM-DD'.

    Returns:
    bool: True if the first date is ahead of the second, False otherwise.
    """
    # Convert the date strings to datetime objects
    date_obj1 = datetime.strptime(date_str1, '%Y-%m-%d')
    date_obj2 = datetime.strptime(date_str2, '%Y-%m-%d')

    # Return True if the first date is ahead of the second
    return date_obj1 > date_obj2

def categorize_time_of_day(date_str, latitude, longitude):
    """
    Categorize the given datetime as:
    - 1 if it is before sunrise or within 30 minutes after sunrise,
    - 3 if it is after sunset or within 30 minutes before sunset,
    - 2 otherwise (during daytime but not within 30 minutes of sunrise or sunset).

    Args:
    date_str (str): Datetime string in the format 'YYYY-MM-DD-HH-MM-SS'.
    latitude (float): Latitude of the location.
    longitude (float): Longitude of the location.

    Returns:
    int: 1, 2, or 3 based on the categorization of the time of day.
    """
    # Define the Central Time timezone
    central = pytz.timezone('America/Chicago')

    # Parse the datetime string and localize to Central Time
    date_obj = datetime.strptime(date_str, '%Y-%m-%d-%H-%M-%S')
    date_obj = central.localize(date_obj)

    # Create a location object
    location = LocationInfo(latitude=latitude, longitude=longitude)

    # Get sunrise and sunset times
    s = sun(location.observer, date=date_obj.date(), tzinfo=central)
    print(s)
    # Check the time conditions
    sunrise_plus_30 = s['sunrise'] + timedelta(minutes=30)
    sunset_minus_30 = s['sunset'] - timedelta(minutes=30)

    if date_obj <= s['sunrise'] or date_obj <= sunrise_plus_30:
        return 1
    elif date_obj >= sunset_minus_30 or date_obj >= s['sunset']:
        return 3
    else:
        return 2

# Example usage
# result = categorize_time_of_day('2024-05-15-05-30-00', 40.8257, -73.699722)
# print(result)