package life.jukebox.yoda;

import java.util.concurrent.*;

/**
 * JBWatcher monitors the state of a live Jukebox
 */
public class JBWatcher implements Runnable{
  private ScheduledFuture<?> taskHandle;
  private String jbId;

  public JBWatcher(String jbId) {
    this.jbId = jbId;
  }

  public ScheduledFuture<?> getWatcherHandle(){
    return taskHandle;
  } 

  public void setWatcherHandle(ScheduledFuture<?> taskHandle){
    this.taskHandle = taskHandle;
  }

  /**
   * run() is called by the scheduler at set intervals to check
   * the current song being played on Spotify belonging
   * to the user having a jukebox with JBId.
   */
  public void run(){
    System.out.println("Spotify API call to get current playing track");
  }
}
