somehow lost old notes. sad :(

### simplequad Environment
* [x] subclass of gym.Env
* [ ] has an __init__ that initializes env. variables
    * [x] initializes action_space and observation_space as gym.spaces
    * [ ] seeds the system randomly (necessary?)
    * [ ] initializes state to default
* [x] has step function that takes in action and returns new state, reward
* [x] has reset function that resets env variables
* [ ] has render function that creates a viewer, and a close function that closes it
    