package life.jukebox.yoda;

import java.util.*;
import java.util.concurrent.*;

/**
 * JBWatcherManager manages all active jukebox watchers.
 */
public class JBWatcherManager {
  private ScheduledExecutorService scheduler;
  private Set<JBWatcher> activeWatchers;
  private static JBWatcherManager manager;
  private static final long WATCHER_PERIOD = 10;

  protected JBWatcherManager() {
    scheduler = Executors.newScheduledThreadPool(4);
    activeWatchers = new HashSet<JBWatcher>();
  }

  public static JBWatcherManager getInstance() {
    if (manager == null) {
      manager = new JBWatcherManager();
    }
    return manager;
  }

  /**
   * Adds a watcher object to the activeWatchers set
   * and creates a taskHandle which schedules a task
   * for watcher reccuring according to WATCHER_PERIOD.
   */
  public void activateWatcher(JBWatcher watcher) {
    if (!activeWatchers.add(watcher)) return;
    ScheduledFuture<?> taskHandle = scheduler.scheduleAtFixedRate(watcher, 0, WATCHER_PERIOD, TimeUnit.SECONDS);
    watcher.setWatcherHandle(taskHandle);
  }

  public void killWatcher(JBWatcher watcher) {
    if (!activeWatchers.remove(watcher)) return;
    watcher.getWatcherHandle().cancel(false);
  }
  
  public boolean isWatcherActive(JBWatcher watcher) {
    return activeWatchers.contains(watcher);
  }
}
