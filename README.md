# Jukebox
[![Build Status](https://travis-ci.org/PedramPejman/jukebox.svg?branch=master)](https://travis-ci.org/PedramPejman/jukebox)

## Purpose

Two phones make a spkear.

## Architecture Overview
This application has two components: The Universe and Jukebox. The Universe Service maintains the central state that allows Jukebox instaces to synchronize playback with oneanother.

Universe (uni): a collection of python and java functions written to be deployed in a serverless environment. Universe leads the syncronization process when two or more Jukeboxes want to connect. In the future, this may need to become a full-fledged service.

Jukebox (juke): A web-native application that walks users through a simple one-click syncronization step and allows them to form a bond with others around them, becoming a SpeakerGroup. 

## Continuous Deployment
The Jukebox project adheres to principles of continuous deployment. All successful merges into the *master branch* are automatically pushed to [the staging envrionment](http://staging.jukebox.life) and merges into *production branch* automatically pushed to [the production environment](http://jukebox.life). Support for canarying will be added when the beta is released and a fully automated A/B test pipeline is imperative.

## Installation for Development

There are currently 2 services to install: uni and juke

TODO(pedrampejman): Give Docker installation instructions

Now, if you visit http://localhost:8080 on your browser, you should see everything working!

TODO: add different entrypoints so that each container can have the option to be linked to staging and production.

## User Journey
Let's go over the most prominent user journey. 

When Albert visits jukebox.life, he gets to either create a jukebox or join a jukebox. 

If he clicks on create, he's taken to /jukebox-create where he is given an easily communicatable JukeboxId (eg. 2 digit number or a simple, common word) and is directed to /jukebox-play where we either invoke native jukebox application (if exists), or show a web-native page (can be a simple "Install Spotify application to control the Jukebox" message). 

If he clicks on join a jukebox, he is taken to a form to supply the JukeboxId (and "..or create a Jukebox yourself" underneath). 

When multiple jukeboxes are joined onto the same channel, as soon as the leader plays a song, they all start to copy playback and play the song in unison. This is clearly difficult to pull of because of the difficulty associated with the precision in central control that is required. We can achieve this by allowing the central state-keeper to account for its delay in making events happen on the Jukebox client devices. 

As jukebox devices register with a jukebox (identified by a JukeboxId), the server starts to test their latency by conducting timed handshakes and recording when it reaches out, and when the client shakes. It will do this continuously while the Jukebox client screens show "Synchronizing" (and possibly some indicator of signal volatility - the measure that determines whether we have enough data on latency to be able to accurately synchronize behavior across slaves). After volatility of the delays for all joined devices is achieved, the delays have stablized, the use can play a song, up to a second after which that event gets picked up by the secondly request to /jukebox-detect. It's then that a request to /jukebox-schedule is sent and all other /jukebox-detect hitters of that JukeboxId become playback slaves..

